import 'package:flutter_local_gemma/flutter_local_gemma.dart';

/// Demo skills used by the example app chat screen.
///
/// To add a new skill:
/// 1. Define it here as a `GemmaSkill` constant.
/// 2. Include it in [demoSkills].
///
/// For JS skills loaded from assets, use:
/// ```dart
/// final skill = await GemmaJsSkill.fromAsset('assets/skills/my_skill');
/// ```
/// For JS skills loaded from a remote URL, use:
/// ```dart
/// final skill = await GemmaJsSkill.fromUrl('https://example.com/skills/my_skill/');
/// ```

// ─── Dart skills ──────────────────────────────────────────────────────────────

/// Returns the current date and time.
final GemmaDartSkill getTimeSkill = GemmaDartSkill(
  name: 'get_time',
  description: 'Returns the current date and time in ISO 8601 format.',
  parametersSchema: {},
  required: [],
  handler: (_) async => GemmaSkillResult.text(DateTime.now().toIso8601String()),
);

/// Converts Morse code to text or text to Morse.
final GemmaDartSkill morseSkill = GemmaDartSkill(
  name: 'morse_code',
  description:
      'Converts text to Morse code or Morse code to text. '
      'Pass {"text": "HELLO"} to encode, or {"morse": "... --- ..."} to decode.',
  parametersSchema: {
    'text': {'type': 'string', 'description': 'Plain text to encode to Morse'},
    'morse': {
      'type': 'string',
      'description':
          'Morse code string to decode (dots and dashes separated by spaces)',
    },
  },
  required: [],
  handler: (args) async {
    final morseMap = {
      'A': '.-',
      'B': '-...',
      'C': '-.-.',
      'D': '-..',
      'E': '.',
      'F': '..-.',
      'G': '--.',
      'H': '....',
      'I': '..',
      'J': '.---',
      'K': '-.-',
      'L': '.-..',
      'M': '--',
      'N': '-.',
      'O': '---',
      'P': '.--.',
      'Q': '--.-',
      'R': '.-.',
      'S': '...',
      'T': '-',
      'U': '..-',
      'V': '...-',
      'W': '.--',
      'X': '-..-',
      'Y': '-.--',
      'Z': '--..',
      '0': '-----',
      '1': '.----',
      '2': '..---',
      '3': '...--',
      '4': '....-',
      '5': '.....',
      '6': '-....',
      '7': '--...',
      '8': '---..',
      '9': '----.',
      ' ': '/',
    };
    final reverseMap = {for (final e in morseMap.entries) e.value: e.key};

    if (args['text'] != null) {
      final text = (args['text'] as String).toUpperCase();
      final encoded = text.split('').map((c) => morseMap[c] ?? '?').join(' ');
      return GemmaSkillResult.text('Morse: $encoded');
    } else if (args['morse'] != null) {
      final morse = args['morse'] as String;
      final words = morse.split(' / ');
      final decoded = words
          .map((word) {
            return word.split(' ').map((c) => reverseMap[c] ?? '?').join('');
          })
          .join(' ');
      return GemmaSkillResult.text('Decoded: $decoded');
    }
    return GemmaSkillResult.text(
      'Please provide either "text" or "morse" argument.',
    );
  },
);

// ─── Inline JS skills ──────────────────────────────────────────────────────────

/// Calculates the sum of two numbers.
final GemmaJsSkill calculatorSkill = GemmaJsSkill(
  name: 'calculate_sum',
  description:
      'Calculates the sum of two numbers. Pass {"a": <number>, "b": <number>}.',
  instructions:
      'Use this tool when you need to add two numbers together. '
      'Arguments should be {"a": 15, "b": 27}.',
  parametersSchema: {
    'type': 'object',
    'properties': {
      'a': {'type': 'number', 'description': 'The first number'},
      'b': {'type': 'number', 'description': 'The second number'},
    },
  },
  scriptHtml: '''
    <script>
      window.ai_edge_gallery_get_result = async (rawArgs) => {
        const args = JSON.parse(rawArgs);
        const a = parseFloat(args.a) || 0;
        const b = parseFloat(args.b) || 0;
        return { result: a + b, operation: `\${a} + \${b} = \${a + b}` };
      };
    </script>
  ''',
);

/// Converts values between common units.
final GemmaJsSkill unitConverterSkill = GemmaJsSkill(
  name: 'convert_units',
  description:
      'Converts a value between common units. Pass {"value": <number>, "from": "<unit>", "to": "<unit>"}.',
  instructions:
      'Use this tool to convert between units. Supported: km/miles, kg/lbs, celsius/fahrenheit. '
      'Example: {"value": 100, "from": "km", "to": "miles"}.',
  parametersSchema: {
    'type': 'object',
    'properties': {
      'value': {'type': 'number', 'description': 'The value to convert'},
      'from': {'type': 'string', 'description': 'The unit to convert from'},
      'to': {'type': 'string', 'description': 'The unit to convert to'},
    },
  },
  scriptHtml: '''
    <script>
      window.ai_edge_gallery_get_result = async (rawArgs) => {
        const args = JSON.parse(rawArgs);
        const val = parseFloat(args.value) || 0;
        const from = (args.from || '').toLowerCase();
        const to   = (args.to   || '').toLowerCase();
        const conversions = {
          'km_miles':          v => v * 0.621371,
          'miles_km':          v => v * 1.60934,
          'kg_lbs':            v => v * 2.20462,
          'lbs_kg':            v => v / 2.20462,
          'celsius_fahrenheit': v => v * 9/5 + 32,
          'fahrenheit_celsius': v => (v - 32) * 5/9,
        };
        const key = `\${from}_\${to}`;
        const fn  = conversions[key];
        if (!fn) return { error: `Unsupported conversion: \${from} to \${to}` };
        const result = fn(val);
        return { result: parseFloat(result.toFixed(4)), from, to };
      };
    </script>
  ''',
);

/// Evaluates a JavaScript expression and returns the result.
final GemmaJsSkill jsEvalSkill = GemmaJsSkill(
  name: 'evaluate_expression',
  description:
      'Evaluates a mathematical or JavaScript expression safely and returns the result. '
      'Pass {"expression": "Math.sqrt(16) + 2"}.',
  instructions:
      'Use this tool to evaluate complex mathematical expressions. '
      'Example arguments: {"expression": "Math.sqrt(144) * 2"}.',
  parametersSchema: {
    'type': 'object',
    'properties': {
      'expression': {
        'type': 'string',
        'description': 'The JavaScript math expression to evaluate',
      },
    },
  },
  scriptHtml: '''
    <script>
      window.ai_edge_gallery_get_result = async (rawArgs) => {
        const args = JSON.parse(rawArgs);
        try {
          const expr = args.expression || '';
          // Safe evaluation using Function constructor (sandboxed in iframe)
          const fn = new Function('Math', 'return (' + expr + ')');
          const result = fn(Math);
          return { result: result, expression: expr };
        } catch(e) {
          return { error: e.message, expression: args.expression };
        }
      };
    </script>
  ''',
);

/// Generates a color palette from a seed color.
final GemmaJsSkill colorPaletteSkill = GemmaJsSkill(
  name: 'generate_color_palette',
  description:
      'Generates a harmonious color palette from a seed color. '
      'Pass {"color": "#FF5733"} or {"color": "blue"}.',
  instructions:
      'Use this tool to generate color palettes for design work. '
      'Returns hex color codes. Example: {"color": "#3498db"}.',
  parametersSchema: {
    'type': 'object',
    'properties': {
      'color': {
        'type': 'string',
        'description':
            'The base color to generate a palette from (e.g. #FF0000)',
      },
    },
  },
  scriptHtml: '''
    <script>
      window.ai_edge_gallery_get_result = async (rawArgs) => {
        const args = JSON.parse(rawArgs);
        const color = args.color || '#3498db';
        // Parse hex to HSL and generate palette
        const hexToRgb = hex => {
          const r = /^#?([a-f\\d]{2})([a-f\\d]{2})([a-f\\d]{2})\$/i.exec(hex.startsWith('#') ? hex : '#' + hex);
          return r ? { r: parseInt(r[1],16), g: parseInt(r[2],16), b: parseInt(r[3],16) } : { r: 52, g: 152, b: 219 };
        };
        const rgbToHsl = ({r, g, b}) => {
          r /= 255; g /= 255; b /= 255;
          const max = Math.max(r,g,b), min = Math.min(r,g,b);
          let h, s, l = (max+min)/2;
          if (max===min) { h = s = 0; }
          else {
            const d = max - min;
            s = l > 0.5 ? d/(2-max-min) : d/(max+min);
            if (max===r) h = ((g-b)/d + (g<b?6:0))/6;
            else if (max===g) h = ((b-r)/d+2)/6;
            else h = ((r-g)/d+4)/6;
          }
          return { h: Math.round(h*360), s: Math.round(s*100), l: Math.round(l*100) };
        };
        const hslToHex = (h,s,l) => {
          s /= 100; l /= 100;
          const a = s * Math.min(l, 1-l);
          const f = n => { const k = (n+h/30)%12; const c = l-a*Math.max(-1,Math.min(k-3,9-k,1)); return Math.round(255*c).toString(16).padStart(2,'0'); };
          return '#' + f(0) + f(8) + f(4);
        };
        const { h, s, l } = rgbToHsl(hexToRgb(color));
        const palette = [
          hslToHex(h, s, Math.max(10, l - 30)),
          hslToHex(h, s, Math.max(10, l - 15)),
          hslToHex(h, s, l),
          hslToHex(h, s, Math.min(90, l + 15)),
          hslToHex(h, s, Math.min(90, l + 30)),
        ];
        return { palette, base: '#' + (color.startsWith('#') ? color.slice(1) : color), hsl: { h, s, l } };
      };
    </script>
  ''',
);

// ─── Asset-based JS skills (loaded from bundled files) ────────────────────────

/// Creates a [GemmaJsSkill] for the QR-code skill bundled in assets.
/// Returns null if the asset is not available.
Future<GemmaJsSkill?> loadQrCodeSkill() async {
  try {
    return await GemmaJsSkill.fromAsset('assets/skills/qr-code');
  } catch (_) {
    return null;
  }
}

/// Creates a [GemmaJsSkill] for the calculator skill bundled in assets.
Future<GemmaJsSkill?> loadCalculatorAssetSkill() async {
  try {
    return await GemmaJsSkill.fromAsset('assets/skills/calculator');
  } catch (_) {
    return null;
  }
}

/// Creates a [GemmaJsSkill] for the Wikipedia query skill bundled in assets.
Future<GemmaJsSkill?> loadWikipediaSkill() async {
  try {
    return await GemmaJsSkill.fromAsset('assets/skills/query-wikipedia');
  } catch (_) {
    return null;
  }
}

// ─── Remote JS skills ────────────────────────────────────────────────────────

/// Loads the hash calculator skill from the AI Edge Gallery.
/// Skips silently (returns null) if the network is unavailable.
Future<GemmaJsSkill?> loadRemoteHashSkill() async {
  try {
    return await GemmaJsSkill.fromUrl(
      'https://raw.githubusercontent.com/google-ai-edge/gallery/refs/heads/main/skills/built-in/calculate-hash/',
    );
  } catch (_) {
    return null;
  }
}

// ─── Bundled skill list ────────────────────────────────────────────────────────

/// Builds the full list of demo skills, including asset-based and remote ones.
///
/// Call this at app startup and await it; some skills require async loading.
/// Falls back gracefully if assets are missing or the network is unavailable.
///
/// ```dart
/// final skills = await buildDemoSkills();
/// ```
Future<List<GemmaSkill>> buildDemoSkills() async {
  final skills = <GemmaSkill>[
    // ── Always available (no I/O) ──────────────────────────────────────────
    getTimeSkill,
    morseSkill,
    calculatorSkill,
    unitConverterSkill,
    jsEvalSkill,
    colorPaletteSkill,
  ];

  // ── Asset-based skills ─────────────────────────────────────────────────
  final qr = await loadQrCodeSkill();
  if (qr != null) skills.add(qr);

  final wiki = await loadWikipediaSkill();
  if (wiki != null) skills.add(wiki);

  try {
    final testSecret = await GemmaJsSkill.fromAsset(
      'assets/skills/test-secret',
      secret: 'MY_SUPER_SECRET_KEY',
    );
    skills.add(testSecret);
  } catch (_) {
    // Ignore if not present
  }

  // ── Remote skills (skipped on offline devices) ─────────────────────────
  final hash = await loadRemoteHashSkill();
  if (hash != null) skills.add(hash);

  return skills;
}

/// Synchronous minimal skill list (no async I/O required).
/// Use this for quick access when `buildDemoSkills()` hasn't completed yet,
/// or when testing skills that don't need asset loading.
final List<GemmaSkill> demoSkills = [
  getTimeSkill,
  morseSkill,
  calculatorSkill,
  unitConverterSkill,
  jsEvalSkill,
  colorPaletteSkill,
];