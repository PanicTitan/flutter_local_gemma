// lib/gemma/schema.dart

/// A Zod-like JSON Schema builder for structured AI outputs.
///
/// Build a schema using the static factory methods, then pass it to
/// [GemmaChat.sendMessageJsonStream] or [SingleTurnChat.generateJson] to
/// instruct the model to produce JSON conforming to that shape.
///
/// ## Example
/// ```dart
/// final schema = Schema.object({
///   'name':  Schema.string().description('Full name of the person'),
///   'age':   Schema.number(),
///   'roles': Schema.array(items: Schema.stringEnum(['admin', 'user'])),
///   'bio':   Schema.string().optional(),
/// });
///
/// final result = await chat.generateJson('Extract: "Alice, 30, admin"', schema: schema);
/// ```
abstract class Schema {
  bool _isOptional = false;
  String? _description;

  /// Marks this field as optional in the generated JSON Schema `required` array.
  Schema optional() {
    _isOptional = true;
    return this;
  }

  /// Adds a `description` hint to the JSON Schema, which improves model adherence.
  Schema description(String desc) {
    _description = desc;
    return this;
  }

  /// Serialises this schema node to a JSON Schema-compatible map.
  Map<String, dynamic> toJsonSchema() {
    final map = build();
    if (_description != null) map['description'] = _description;
    return map;
  }

  /// Subclasses implement this to return the core schema map (without description).
  Map<String, dynamic> build();

  /// Whether [optional] has been called on this node.
  bool get isOptional => _isOptional;

  // ── Factory constructors ─────────────────────────────────────────────────

  /// A JSON string field.
  static StringSchema string() => StringSchema();

  /// A JSON number field (integer or float).
  static NumberSchema number() => NumberSchema();

  /// A JSON boolean field.
  static BooleanSchema boolean() => BooleanSchema();

  /// A JSON object with the given named [properties].
  ///
  /// Properties that are not marked [optional] are added to `required`.
  /// `additionalProperties` is set to `false` to keep the model from adding
  /// extra fields.
  static ObjectSchema object(Map<String, Schema> properties) =>
      ObjectSchema(properties);

  /// A JSON array whose elements conform to [items].
  static ArraySchema array({required Schema items}) => ArraySchema(items);

  /// A string field constrained to one of the given [values] (JSON `enum`).
  static EnumSchema stringEnum(List<String> values) => EnumSchema(values);
}

/// Schema node that maps to JSON `"type": "string"`.
class StringSchema extends Schema {
  @override
  Map<String, dynamic> build() => {'type': 'string'};
}

/// Schema node that maps to JSON `"type": "number"`.
class NumberSchema extends Schema {
  @override
  Map<String, dynamic> build() => {'type': 'number'};
}

/// Schema node that maps to JSON `"type": "boolean"`.
class BooleanSchema extends Schema {
  @override
  Map<String, dynamic> build() => {'type': 'boolean'};
}

/// Schema node that maps to a JSON `"type": "object"` with named properties.
class ObjectSchema extends Schema {
  /// The named property schemas for this object.
  final Map<String, Schema> properties;
  ObjectSchema(this.properties);

  @override
  Map<String, dynamic> build() {
    final requiredProps = properties.entries
        .where((e) => !e.value.isOptional)
        .map((e) => e.key)
        .toList();

    return {
      'type': 'object',
      'properties': properties.map((k, v) => MapEntry(k, v.toJsonSchema())),
      if (requiredProps.isNotEmpty) 'required': requiredProps,
      'additionalProperties': false,
    };
  }
}

/// Schema node that maps to a JSON `"type": "array"` with a typed [items] schema.
class ArraySchema extends Schema {
  /// The schema for each element of the array.
  final Schema items;
  ArraySchema(this.items);

  @override
  Map<String, dynamic> build() => {
    'type': 'array',
    'items': items.toJsonSchema(),
  };
}

/// Schema node that constrains a string to a fixed set of [values] (JSON `enum`).
class EnumSchema extends Schema {
  /// The allowed string values.
  final List<String> values;
  EnumSchema(this.values);

  @override
  Map<String, dynamic> build() => {'type': 'string', 'enum': values};
}
