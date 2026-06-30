@file:Suppress("OPT_IN_USAGE")
package com.example.flutter_local_gemma

import android.util.Log
import com.google.ai.edge.litertlm.Tool
import com.google.ai.edge.litertlm.ToolParam
import com.google.ai.edge.litertlm.ToolSet
import kotlinx.coroutines.runBlocking
import org.json.JSONObject

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.suspendCancellableCoroutine
import io.flutter.plugin.common.MethodChannel

private const val TAG = "GemmaAgentToolSet"

/**
 * Native ToolSet for LiteRT-LM SDK.
 *
 * Routes tool executions natively from the C++ LLM Engine directly to Dart
 * via a Flutter MethodChannel.
 */
class GemmaAgentToolSet(
    private val jsSkillExecutor: JsSkillExecutor,
    private val methodChannel: MethodChannel? = null
) : ToolSet {

    data class SkillDef(
        val name: String,
        val description: String,
        val instructions: String,
        val scriptHtml: String?,
        val timeoutMs: Long = 30_000L,
    )

    var registeredSkills: List<SkillDef> = emptyList()

    @Tool(description = "Loads a skill's instructions. Always call this before run_js.")
    fun load_skill(
        @ToolParam(description = "The exact skill name to load.") skillName: String
    ): Map<String, String> {
        val skill = registeredSkills.find { it.name == skillName.trim() }
        return if (skill != null) {
            val content = "---\nname: ${skill.name}\ndescription: ${skill.description}\n---\n\n${skill.instructions}"
            Log.d(TAG, "load_skill OK: '$skillName'")
            mapOf("skill_name" to skillName, "skill_instructions" to content)
        } else {
            Log.w(TAG, "load_skill: '$skillName' not found")
            val available = registeredSkills.joinToString { "\"${it.name}\"" }
            mapOf("error" to "Skill '$skillName' not found. Available: $available")
        }
    }

    @Tool(description = "Runs a JS script for a skill. Always call load_skill first.")
    fun run_js(
        @ToolParam(description = "The skill name.") skillName: String,
        @ToolParam(description = "Script filename. Use 'index.html' unless specified.") scriptName: String,
        @ToolParam(description = "A stringified JSON representing the data for the script. Example: '{\"key\": \"value\"}'") data: String,
    ): Map<String, Any> {
        val dataString = data
        Log.d(TAG, "run_js: skill='$skillName' script='$scriptName' data='${dataString.take(100)}'")

        val skill = registeredSkills.find { it.name == skillName.trim() }
            ?: return mapOf("error" to "Skill '$skillName' not found", "status" to "failed")

        val html = skill.scriptHtml
            ?: return mapOf("error" to "Skill '$skillName' is text-only (no JS script)", "status" to "failed")

        return runBlocking {
            try {
                val raw = jsSkillExecutor.execute(
                    skillName  = skillName,
                    scriptName = scriptName,
                    scriptHtml = html,
                    argsJson   = dataString.trim().ifEmpty { "{}" },
                    timeoutMs  = skill.timeoutMs,
                )
                val json = JSONObject(raw)
                val error = json.optString("error", "").takeIf { it.isNotEmpty() }
                if (error != null) {
                    mapOf("error" to error, "status" to "failed")
                } else {
                    mapOf("result" to json.optString("result", ""), "status" to "succeeded")
                }
            } catch (e: Exception) {
                Log.e(TAG, "run_js error for '$skillName'", e)
                mapOf("error" to (e.message ?: "Unknown"), "status" to "failed")
            }
        }
    }

    @Tool(description = "Executes a system skill or custom function.")
    fun execute_skill(
        @ToolParam(description = "The exact skill name to execute.") skillName: String,
        @ToolParam(description = "A stringified JSON representation of the arguments. Example: '{\"key\": \"value\"}'") argsJson: String
    ): Map<String, Any> {
        val argsString = argsJson
        Log.d(TAG, "execute_skill: skill='$skillName' args='$argsString'")
        
        if (methodChannel == null) {
            return mapOf("error" to "MethodChannel not attached", "status" to "failed")
        }

        return runBlocking {
            try {
                // Parse args string to Map so Dart receives proper structure
                val argsMap = try {
                    val json = JSONObject(argsString)
                    val map = mutableMapOf<String, Any>()
                    json.keys().forEach { key -> map[key] = json.get(key) }
                    map
                } catch (e: Exception) {
                    emptyMap<String, Any>()
                }

                val result = withContext(Dispatchers.Main) {
                    suspendCancellableCoroutine<Any?> { continuation ->
                        methodChannel.invokeMethod("executeTool", mapOf(
                            "skillName" to skillName,
                            "args" to argsMap
                        ), object : MethodChannel.Result {
                            override fun success(result: Any?) {
                                continuation.resume(result)
                            }

                            override fun error(errorCode: String, errorMessage: String?, errorDetails: Any?) {
                                continuation.resumeWithException(Exception("$errorCode: $errorMessage"))
                            }

                            override fun notImplemented() {
                                continuation.resumeWithException(Exception("Not implemented"))
                            }
                        })
                    }
                }
                
                // Return Dart's result, wrapping it appropriately
                @Suppress("UNCHECKED_CAST")
                if (result is Map<*, *>) {
                    result.entries
                        .filter { it.key != null }
                        .associate { it.key.toString() to (it.value ?: "") }
                } else {
                    mapOf("result" to (result?.toString() ?: ""), "status" to "succeeded")
                }
            } catch (e: Exception) {
                Log.e(TAG, "execute_skill error for '$skillName'", e)
                mapOf("error" to (e.message ?: "Unknown"), "status" to "failed")
            }
        }
    }
}