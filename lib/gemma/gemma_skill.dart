import 'dart:convert';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/services.dart' show rootBundle;
import '../utils/http_fetch.dart';

/// Gallery-compatible result format for skill execution.
class GemmaSkillResult {
  final String result;
  final String? imageBase64;
  final GemmaWebviewResult? webview;

  const GemmaSkillResult.text(this.result) : imageBase64 = null, webview = null;
  const GemmaSkillResult.withImage(this.result, this.imageBase64)
    : webview = null;
  const GemmaSkillResult.withWebview(this.result, this.webview)
    : imageBase64 = null;
  const GemmaSkillResult._({
    required this.result,
    this.imageBase64,
    this.webview,
  });

  factory GemmaSkillResult.fromGalleryJson(Map<String, dynamic> json) {
    return GemmaSkillResult._(
      result: json['result'] as String? ?? '',
      imageBase64:
          (json['image'] as Map<String, dynamic>?)?['base64'] as String?,
      webview: json['webview'] != null
          ? GemmaWebviewResult.fromJson(json['webview'] as Map<String, dynamic>)
          : null,
    );
  }

  Map<String, dynamic> toGalleryJson() => {
    'result': result,
    if (imageBase64 != null) 'image': {'base64': imageBase64},
    if (webview != null) 'webview': webview!.toJson(),
    'error': null,
  };
}

class GemmaWebviewResult {
  final String url;
  final bool iframe;
  final double aspectRatio;

  const GemmaWebviewResult({
    required this.url,
    this.iframe = true,
    this.aspectRatio = 1.33,
  });

  factory GemmaWebviewResult.fromJson(Map<String, dynamic> json) =>
      GemmaWebviewResult(
        url: json['url'] as String,
        iframe: json['iframe'] as bool? ?? true,
        aspectRatio: (json['aspectRatio'] as num?)?.toDouble() ?? 1.33,
      );

  Map<String, dynamic> toJson() => {
    'url': url,
    'iframe': iframe,
    'aspectRatio': aspectRatio,
  };
}

// ─── Sealed skill base ────────────────────────────────────────────────────────

/// Abstract base class for all Gemma skills.
///
/// Both [GemmaDartSkill] (Dart-native handler) and [GemmaJsSkill]
/// (JS executed in WebView/iframe) extend this class.
///
/// Use [GemmaSkill] wherever you previously used `Object` in skill lists:
/// ```dart
/// final List<GemmaSkill> skills = [myDartSkill, myJsSkill];
/// ```
///
/// `List<Object>` is still accepted in all APIs for backwards compatibility.
sealed class GemmaSkill {
  const GemmaSkill();

  /// Unique identifier used in tool-call JSON (`"name": "..."`)
  String get name;

  /// Human-readable description shown to the model in the system prompt.
  String get description;
}

/// Convenience typedef for a typed skill list.
typedef GemmaSkillList = List<GemmaSkill>;

// ─── Dart skill ───────────────────────────────────────────────────────────────

/// A Dart-native skill whose [handler] runs directly in the Dart isolate.
///
/// ```dart
/// final timeSkill = GemmaDartSkill(
///   name: 'get_time',
///   description: 'Returns the current date and time in ISO 8601 format.',
///   parametersSchema: {},
///   handler: (args) async => GemmaSkillResult.text(DateTime.now().toIso8601String()),
/// );
/// ```
final class GemmaDartSkill extends GemmaSkill {
  @override
  final String name;
  @override
  final String description;

  /// JSON Schema `properties` object for this skill's input parameters.
  final Map<String, dynamic> parametersSchema;

  /// Required parameter names (subset of [parametersSchema] keys).
  final List<String> required;

  /// The handler function. Receives parsed arguments and returns a [GemmaSkillResult].
  final Future<GemmaSkillResult> Function(Map<String, dynamic> args) handler;

  const GemmaDartSkill({
    required this.name,
    required this.description,
    required this.parametersSchema,
    this.required = const [],
    required this.handler,
  });
}

// ─── JS skill ─────────────────────────────────────────────────────────────────

/// A JavaScript-based skill that runs inside a WebView pool on Android or an
/// iframe sandbox on Web.
///
/// JS skills follow the Google AI Edge Gallery format:
/// - `SKILL.md` — YAML frontmatter with `name` and `description`, followed by
///   natural-language instructions for the model.
/// - `scripts/index.html` — exposes `window.ai_edge_gallery_get_result(data)`
///   which receives a JSON string and returns a JSON string.
///
/// Load from assets or a remote URL:
/// ```dart
/// final skill = await GemmaJsSkill.fromAsset('assets/skills/calculator');
/// final skill = await GemmaJsSkill.fromUrl('https://example.com/skills/qr-code');
/// ```
final class GemmaJsSkill extends GemmaSkill {
  @override
  final String name;
  @override
  final String description;

  /// Natural-language instructions shown to the model for how to call this skill.
  final String instructions;

  /// The raw HTML (including `<script>` tag) that implements the skill.
  final String scriptHtml;

  /// JSON Schema `properties` object for this skill's input parameters.
  final Map<String, dynamic> parametersSchema;

  /// Optional secret (API key, token, etc.) injected into the JS execution sandbox.
  final String? secret;

  const GemmaJsSkill({
    required this.name,
    required this.description,
    required this.instructions,
    required this.scriptHtml,
    this.parametersSchema = const {},
    this.secret,
  });

  /// Loads a JS skill bundled as a Flutter asset.
  ///
  /// [assetDirPath] should point to a directory containing `SKILL.md` and
  /// `scripts/index.html`. Example: `'assets/skills/calculator'`.
  static Future<GemmaJsSkill> fromAsset(String assetDirPath, {String? secret}) async {
    final skillMd = await rootBundle.loadString('$assetDirPath/SKILL.md');
    final indexHtml = await rootBundle.loadString(
      '$assetDirPath/scripts/index.html',
    );
    final (name, desc, instructions, schema) = _parseSkillMd(skillMd);
    return GemmaJsSkill(
      name: name,
      description: desc,
      instructions: instructions,
      scriptHtml: indexHtml,
      parametersSchema: schema,
      secret: secret,
    );
  }

  /// Loads a JS skill from a remote URL.
  ///
  /// [baseUrl] should point to the skill root directory (no trailing slash needed).
  /// Files fetched: `SKILL.md` and `scripts/index.html`.
  static Future<GemmaJsSkill> fromUrl(String baseUrl, {String? secret}) async {
    final base = baseUrl.endsWith('/') ? baseUrl : '$baseUrl/';
    final skillMd = await fetchText('${base}SKILL.md');
    final indexHtml = await fetchText('${base}scripts/index.html');
    final (name, desc, instructions, schema) = _parseSkillMd(skillMd);

    // Remote scripts (like from GitHub) can be blocked by strict MIME type checking
    // or fail to load on Android because the WebView uses a local asset base URL.
    // To bypass this, we find all `<script src="...">` tags, fetch their content manually,
    // and inline them directly into the HTML string.
    String finalHtml = indexHtml;
    final scriptRe = RegExp(r'<script\s+src="([^"]+)">\s*</script>', caseSensitive: false);
    while (true) {
      final match = scriptRe.firstMatch(finalHtml);
      if (match == null) break;
      final src = match.group(1)!;
      try {
        final scriptContent = await fetchText('${base}scripts/$src');
        finalHtml = finalHtml.replaceFirst(match.group(0)!, '<script>\n$scriptContent\n</script>');
      } catch (e) {
        // If fetch fails, replace with a comment to avoid infinite loop
        finalHtml = finalHtml.replaceFirst(match.group(0)!, '<!-- failed to load $src -->');
      }
    }

    return GemmaJsSkill(
      name: name,
      description: desc,
      instructions: instructions,
      scriptHtml: finalHtml,
      parametersSchema: schema,
      secret: secret,
    );
  }

  static (String, String, String, Map<String, dynamic>) _parseSkillMd(
    String md,
  ) {
    final frontmatterRe = RegExp(r'^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$');
    final m = frontmatterRe.firstMatch(md.trim());
    final fm = m?.group(1) ?? '';
    final body = m?.group(2)?.trim() ?? '';
    final nameMatch = RegExp(r'^name:\s*(.+)$', multiLine: true).firstMatch(fm);
    final descMatch = RegExp(
      r'^description:\s*(.+)$',
      multiLine: true,
    ).firstMatch(fm);

    Map<String, dynamic> schema = {};
    final schemaMatch = RegExp(
      r'^parameters:\s*([\s\S]+)$',
      multiLine: true,
    ).firstMatch(fm);
    if (schemaMatch != null) {
      try {
        schema = jsonDecode(schemaMatch.group(1)!) as Map<String, dynamic>;
      } catch (_) {}
    }

    return (
      nameMatch?.group(1)?.trim() ?? '',
      descMatch?.group(1)?.trim() ?? '',
      body,
      schema,
    );
  }
}

/// Builds the `___SKILLS___` section for the JSON prompt injection.
String buildPromptSkillsSection(List<Object> skills) {
  return skills
      .map((s) {
        if (s is GemmaDartSkill) {
          final jsonDesc = jsonEncode({
            "name": s.name,
            "description": s.description,
            "parameters": s.parametersSchema,
          });
          return '<|tool>\n$jsonDesc\n<tool|>';
        } else if (s is GemmaJsSkill) {
          final jsonDescMap = <String, dynamic>{
            "name": s.name,
            "description": s.description,
          };
          if (s.parametersSchema.isNotEmpty) {
            jsonDescMap["parameters"] = s.parametersSchema;
          }
          final jsonDesc = jsonEncode(jsonDescMap);
          return '<|tool>\n$jsonDesc\n<tool|>';
        }
        return '';
      })
      .join('\n\n');
}

/// Builds the native tool declarations for Gemma 4.
String buildNativeSkillsSection(List<Object> skills) {
  return skills
      .map((s) {
        if (s is GemmaDartSkill) {
          // In native mode, we format parameters without JSON braces using the custom format.
          // E.g. parameters:{location:{type:<|"|>string<|"|>,description:<|"|>The city name<|"|>}}
          // For simplicity if parametersSchema is empty, we omit parameters.
          final properties = s.parametersSchema['properties'];
          final props = properties is Map
              ? properties.cast<String, dynamic>()
              : <String, dynamic>{};
          String paramsStr = '';
          if (props.isNotEmpty) {
            final propsList = props.entries
                .map((e) {
                  final type = e.value['type'] ?? 'string';
                  final desc = e.value['description'] ?? '';
                  return '${e.key}:{type:<|"|>$type<|"|>,description:<|"|>$desc<|"|>}';
                })
                .join(',');
            paramsStr = ',parameters:{$propsList}';
          }
          return '<|tool>declaration:${s.name}{description:<|"|>${s.description}<|"|>$paramsStr}<tool|>';
        } else if (s is GemmaJsSkill) {
          final properties = s.parametersSchema['properties'];
          final props = properties is Map
              ? properties.cast<String, dynamic>()
              : <String, dynamic>{};
          String paramsStr = '';
          if (props.isNotEmpty) {
            final propsList = props.entries
                .map((e) {
                  final type = e.value['type'] ?? 'string';
                  final desc = e.value['description'] ?? '';
                  return '${e.key}:{type:<|"|>$type<|"|>,description:<|"|>$desc<|"|>}';
                })
                .join(',');
            paramsStr = ',parameters:{$propsList}';
          }
          return '<|tool>declaration:${s.name}{description:<|"|>${s.description}<|"|>$paramsStr}<tool|>';
        }
        return '';
      })
      .join('');
}