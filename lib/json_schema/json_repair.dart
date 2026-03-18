// lib/gemma/json_repair.dart

/// Streaming-tolerant JSON repair for partial or malformed LLM output.
///
/// LLMs frequently return:
/// - Markdown code fences (` ```json … ``` `)
/// - Unquoted keys
/// - Trailing commas
/// - Truncated JSON (the generation was cut off mid-stream)
///
/// [repairJson] handles all of these cases.  When [streamStable] is `true`
/// it also accepts valid prefixes (incomplete objects / arrays), which makes
/// it suitable for streaming UI updates — each token arrival can be passed
/// through and a best-effort partial object is returned for live rendering.
///
/// ## Basic usage
/// ```dart
/// final raw = '```json\n{"name": "Alice", "age": 30}\n```';
/// final parsed = repairJson(raw); // {name: Alice, age: 30}
/// ```
///
/// ## Streaming usage
/// ```dart
/// final buffer = StringBuffer();
/// await for (final chunk in session.generateResponseStream(parts)) {
///   buffer.write(chunk);
///   final partial = repairJson(buffer.toString(), streamStable: true);
///   if (partial != null) setState(() => _preview = partial);
/// }
/// ```

import 'dart:convert';
import 'dart:math';
import 'package:collection/collection.dart';

typedef JSONReturnType = dynamic;
const List<String> stringDelimiters = ['"', "'", "“", "”"];

enum ContextValues { objectKey, objectValue, array }

class JsonContext {
  final List<ContextValues> _context = [];
  ContextValues? _current;
  bool _empty = true;

  ContextValues? get current => _current;
  bool get isEmpty => _empty;
  List<ContextValues> get context => _context;

  void set(ContextValues value) {
    _context.add(value);
    _current = value;
    _empty = false;
  }

  void reset() {
    if (_context.isNotEmpty) {
      _context.removeLast();
      if (_context.isNotEmpty) {
        _current = _context.last;
      } else {
        _current = null;
        _empty = true;
      }
    } else {
      _current = null;
      _empty = true;
    }
  }
}

class ObjectComparer {
  static bool isSameObject(dynamic obj1, dynamic obj2) {
    if (obj1.runtimeType != obj2.runtimeType) return false;
    if (obj1 is Map) {
      if (obj2 is! Map) return false;
      return const MapEquality().equals(obj1, obj2);
    }
    if (obj1 is List) {
      if (obj2 is! List) return false;
      return const ListEquality().equals(obj1, obj2);
    }
    return obj1 == obj2;
  }

  static bool isStrictlyEmpty(dynamic value) {
    if (value is String || value is List || value is Map || value is Set) {
      return (value as dynamic).isEmpty;
    }
    return false;
  }
}

class InternalParser {
  String jsonStr;
  int index = 0;
  final JsonContext context = JsonContext();
  final bool logging;
  final List<Map<String, String>> logger = [];
  final bool streamStable;

  InternalParser(
    this.jsonStr, {
    this.logging = false,
    this.streamStable = false,
  });

  void _log(String text) {
    if (!logging) return;
    int window = 10;
    int start = max(index - window, 0);
    int end = min(index + window, jsonStr.length);
    logger.add({'text': text, 'context': jsonStr.substring(start, end)});
  }

  JSONReturnType parse() {
    JSONReturnType json = _parseJson();
    if (index < jsonStr.length) {
      _log("Parser returned early, checking for more elements");
      List<JSONReturnType> resultList = [json];
      while (index < jsonStr.length) {
        JSONReturnType j = _parseJson();
        if (j != "") {
          if (resultList.isNotEmpty &&
              ObjectComparer.isSameObject(resultList.last, j)) {
            resultList.removeLast();
          }
          resultList.add(j);
        } else {
          index++;
        }
      }
      if (resultList.length == 1) {
        json = resultList[0];
      } else {
        json = resultList;
      }
    }
    return json;
  }

  JSONReturnType _parseJson() {
    while (index < jsonStr.length) {
      String? char = _getCharAt();
      if (char == null) return "";

      switch (char) {
        case '{':
          index++;
          return _parseObject();
        case '[':
          index++;
          return _parseArray();
      }

      if (!context.isEmpty &&
          (stringDelimiters.contains(char) || _isAlpha(char))) {
        return _parseString();
      } else if (!context.isEmpty &&
          (_isDigit(char) || char == '-' || char == '.')) {
        return _parseNumber();
      } else if (char == '#' || char == '/') {
        return _parseComment();
      } else {
        index++;
      }
    }
    return "";
  }

  JSONReturnType _parseComment() {
    String? char = _getCharAt();
    if (char == null) return "";
    final terms = ['\n', '\r'];
    if (context.context.contains(ContextValues.array)) terms.add(']');
    if (context.context.contains(ContextValues.objectValue)) terms.add('}');
    if (context.context.contains(ContextValues.objectKey)) terms.add(':');

    if (char == '#') {
      while (_getCharAt() != null && !terms.contains(_getCharAt())) index++;
    } else if (char == '/') {
      String? nextChar = _getCharAt(1);
      if (nextChar == '/') {
        index += 2;
        while (_getCharAt() != null && !terms.contains(_getCharAt())) index++;
      } else if (nextChar == '*') {
        index += 2;
        while (_getCharAt() != null) {
          if (_getCharAt(0) == '*' && _getCharAt(1) == '/') {
            index += 2;
            break;
          }
          index++;
        }
      } else {
        index++;
      }
    }
    if (context.isEmpty) return _parseJson();
    return "";
  }

  List<JSONReturnType> _parseArray() {
    List<JSONReturnType> arr = [];
    context.set(ContextValues.array);
    String? char = _getCharAt();

    while (char != null && char != ']' && char != '}') {
      _skipWhitespaces();
      JSONReturnType value = _parseJson();

      if (ObjectComparer.isStrictlyEmpty(value)) {
        _skipWhitespaces();
        if (_getCharAt() == ',') index++;
      } else {
        arr.add(value);
      }

      _skipWhitespaces();
      char = _getCharAt();
      if (char == ',') {
        index++;
        _skipWhitespaces();
        char = _getCharAt();
      }
    }
    if (char == ']') index++;
    context.reset();
    return arr;
  }

  Map<String, JSONReturnType> _parseObject() {
    Map<String, JSONReturnType> obj = {};
    while (_getCharAt() != null && _getCharAt() != '}') {
      _skipWhitespaces();
      if (_getCharAt() == ':') {
        index++;
        continue;
      }

      context.set(ContextValues.objectKey);
      String key = _parseString().toString();
      _skipWhitespaces();
      if (key.isEmpty && _getCharAt() == '}') break;
      if (_getCharAt() == ':') index++;

      context.reset();
      context.set(ContextValues.objectValue);
      _skipWhitespaces();

      JSONReturnType value = (['}', ','].contains(_getCharAt()))
          ? ""
          : _parseJson();
      obj[key] = value;
      context.reset();

      _skipWhitespaces();
      if (_getCharAt() == ',') index++;
    }
    if (_getCharAt() == '}') index++;
    return obj;
  }

  JSONReturnType _parseString() {
    bool missingQuotes = false;
    String rstringDelimiter = '"';
    _skipWhitespaces();
    String? char = _getCharAt();
    if (char == null) return "";

    if (char == '#' || char == '/') return _parseComment();

    if (!stringDelimiters.contains(char)) {
      if (_isAlpha(char)) {
        if (['t', 'f', 'n'].contains(char.toLowerCase()) &&
            context.current != ContextValues.objectKey) {
          final boolOrNull = _parseBooleanOrNull();
          if (boolOrNull != "") return boolOrNull;
        }
        missingQuotes = true;
      } else {
        return "";
      }
    } else {
      rstringDelimiter = {'“': '”', '”': '“', "'": "'", '"': '"'}[char]!;
      index++;
    }

    String stringAcc = "";
    char = _getCharAt();

    while (char != null && char != rstringDelimiter) {
      if (missingQuotes) {
        if (context.current == ContextValues.objectKey &&
            (char == ':' || char.trim().isEmpty))
          break;
        if (context.current == ContextValues.objectValue &&
            [',', '}'].contains(char))
          break;
      }
      if (char == '\\') {
        stringAcc += char;
        index++;
        char = _getCharAt();
        if (char != null) {
          stringAcc += char;
          index++;
          char = _getCharAt();
        }
        continue;
      }
      stringAcc += char;
      index++;
      char = _getCharAt();
    }
    if (char == rstringDelimiter) index++;

    if (missingQuotes || (stringAcc.isNotEmpty && stringAcc.endsWith('\n'))) {
      return stringAcc.trimRight();
    }
    return stringAcc;
  }

  JSONReturnType _parseBooleanOrNull() {
    final startingIndex = index;
    bool? tryParse(String keyword, bool? value) {
      if (jsonStr.length >= startingIndex + keyword.length &&
          jsonStr
                  .substring(startingIndex, startingIndex + keyword.length)
                  .toLowerCase() ==
              keyword) {
        index += keyword.length;
        return value;
      }
      return null;
    }

    String? char = _getCharAt()?.toLowerCase();
    if (char == 't') {
      if (tryParse('true', true) != null) return true;
    } else if (char == 'f') {
      if (tryParse('false', false) != null) return false;
    } else if (char == 'n') {
      if (tryParse('null', null) != null) return null;
    }

    index = startingIndex;
    return "";
  }

  JSONReturnType _parseNumber() {
    final start = index;
    String? char = _getCharAt();
    while (char != null && '0123456789-.eE'.contains(char)) {
      index++;
      char = _getCharAt();
    }
    String numberStr = jsonStr.substring(start, index);
    try {
      if (numberStr.contains('.') ||
          numberStr.contains('e') ||
          numberStr.contains('E')) {
        return double.parse(numberStr);
      } else {
        return int.parse(numberStr);
      }
    } catch (e) {
      return numberStr;
    }
  }

  String? _getCharAt([int count = 0]) {
    final pos = index + count;
    if (pos >= 0 && pos < jsonStr.length) return jsonStr[pos];
    return null;
  }

  void _skipWhitespaces() {
    while (index < jsonStr.length && jsonStr[index].trim().isEmpty) index++;
  }

  bool _isAlpha(String char) => char.toLowerCase() != char.toUpperCase();
  bool _isDigit(String char) => '0123456789'.contains(char);
}

/// Automatically parses and repairs a malformed JSON string.
JSONReturnType repairJson(
  String jsonStr, {
  bool skipDecodeAttempt = false,
  bool logging = false,
  bool streamStable = false,
}) {
  // Pre-process: Strip markdown formatting often returned by LLMs
  String cleanStr = jsonStr.trimLeft();
  if (cleanStr.startsWith('```json')) {
    cleanStr = cleanStr.substring(7).trimLeft();
  } else if (cleanStr.startsWith('```')) {
    cleanStr = cleanStr.substring(3).trimLeft();
  }
  if (!streamStable && cleanStr.endsWith('```')) {
    cleanStr = cleanStr.substring(0, cleanStr.length - 3).trimRight();
  }

  JSONReturnType doRepair() {
    final parser = InternalParser(
      cleanStr,
      logging: logging,
      streamStable: streamStable,
    );
    final result = parser.parse();
    if (logging) return {'data': result, 'log': parser.logger};
    return result;
  }

  if (skipDecodeAttempt) return doRepair();

  try {
    final decoded = jsonDecode(cleanStr);
    if (logging) return {'data': decoded, 'log': []};
    return decoded;
  } catch (e) {
    return doRepair();
  }
}
