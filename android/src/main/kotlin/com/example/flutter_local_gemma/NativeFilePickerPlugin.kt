package com.example.flutter_local_gemma

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.util.Log
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.PluginRegistry

/**
 * NativeFilePickerPlugin
 *
 * Presents Android's system file picker (`ACTION_OPEN_DOCUMENT`) and returns the
 * selected file's content-URI string back to Flutter.
 *
 * ## Why a native picker?
 * Flutter's own `file_picker` package cannot request `FLAG_GRANT_PERSISTABLE_URI_PERMISSION`,
 * which is required when the app needs to read a large model file via a content URI on
 * subsequent app launches without copying it to internal storage first.
 *
 * ## Methods
 * | Method     | Returns                       | Description                             |
 * |------------|-------------------------------|-----------------------------------------|
 * | `pickFile` | `String?` (content URI)       | Opens the file picker. Returns null if the user cancels. |
 *
 * ## Threading / Activity lifecycle
 * This plugin is [ActivityAware] because file picking requires an [Activity] context.
 * `pendingResult` is cleared in [onActivityResult] regardless of success or cancellation
 * to prevent result leaks across configuration changes.
 */
class NativeFilePickerPlugin :
    FlutterPlugin,
    MethodChannel.MethodCallHandler,
    ActivityAware,
    PluginRegistry.ActivityResultListener {

    companion object {
        private const val TAG         = "NativeFilePicker"
        private const val CHANNEL     = "native_file_picker"
        private const val REQUEST_CODE = 9991
    }

    // ── Flutter channel ───────────────────────────────────────────────────────
    private lateinit var channel: MethodChannel

    // ── Activity reference (null when detached) ───────────────────────────────
    /**
     * Held weakly via the plugin binding lifecycle; set to null in
     * [onDetachedFromActivity] and [onDetachedFromActivityForConfigChanges].
     */
    private var activity: Activity? = null

    /**
     * The [MethodChannel.Result] waiting for a file selection.
     * Set before `startActivityForResult` and cleared in [onActivityResult].
     * Only one pick operation can be in-flight at a time.
     */
    private var pendingResult: MethodChannel.Result? = null

    // ─────────────────────────────────────────────────────────────────────────
    // FlutterPlugin
    // ─────────────────────────────────────────────────────────────────────────

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, CHANNEL)
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        // If a result is pending (e.g. engine was detached mid-pick), cancel it gracefully.
        pendingResult?.success(null)
        pendingResult = null
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MethodChannel.MethodCallHandler
    // ─────────────────────────────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "pickFile") {
            result.notImplemented()
            return
        }

        val act = activity
        if (act == null) {
            result.error("NO_ACTIVITY", "No active Activity to present the file picker.", null)
            return
        }

        // Guard against overlapping calls (e.g. user taps twice very quickly).
        if (pendingResult != null) {
            result.error("ALREADY_ACTIVE", "A file pick operation is already in progress.", null)
            return
        }

        pendingResult = result

        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "*/*"
        }

        try {
            act.startActivityForResult(intent, REQUEST_CODE)
        } catch (e: Exception) {
            Log.e(TAG, "startActivityForResult failed: ${e.message}", e)
            pendingResult?.error("LAUNCH_ERROR", e.message, null)
            pendingResult = null
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // PluginRegistry.ActivityResultListener
    // ─────────────────────────────────────────────────────────────────────────

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode != REQUEST_CODE) return false

        val pending = pendingResult
        pendingResult = null // always clear before calling result methods

        if (resultCode == Activity.RESULT_OK) {
            val uri: Uri? = data?.data
            if (uri != null) {
                // Request a persistable read permission so the URI can be re-opened
                // after the file picker session ends (needed for large model files).
                try {
                    activity?.contentResolver?.takePersistableUriPermission(
                        uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                    )
                } catch (e: Exception) {
                    // Non-fatal: some URIs (e.g. on certain OEM file systems) do not support
                    // persistable permissions. The URI is still usable for the current session.
                    Log.w(TAG, "Could not take persistable permission: ${e.message}")
                }
                pending?.success(uri.toString())
            } else {
                pending?.success(null)
            }
        } else {
            // User pressed Back or cancelled – return null, not an error.
            pending?.success(null)
        }

        return true
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ActivityAware
    // ─────────────────────────────────────────────────────────────────────────

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activity = binding.activity
        binding.addActivityResultListener(this)
    }

    override fun onDetachedFromActivityForConfigChanges() {
        // Activity is being recreated (e.g. rotation). Clear the reference; it will be
        // reassigned in onReattachedToActivityForConfigChanges.
        activity = null
    }

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {
        activity = binding.activity
        binding.addActivityResultListener(this)
    }

    override fun onDetachedFromActivity() {
        activity = null
    }
}