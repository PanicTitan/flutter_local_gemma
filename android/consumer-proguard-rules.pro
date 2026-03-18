# Keep LiteRT and related classes used by the native plugin
-keep class com.google.ai.edge.litertlm.** { *; }

# Keep Kotlin coroutines internals used by plugin
-keep class kotlinx.coroutines.** { *; }
