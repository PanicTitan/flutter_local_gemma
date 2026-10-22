package com.example.flutter_local_gemma

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.documentfile.provider.DocumentFile
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.PluginRegistry
import kotlinx.coroutines.*

/**
 * NativeFilePickerPlugin
 *
 * Presents Android's system file and directory pickers via Storage Access Framework:
 * - `pickFile`: Single file picker with persistable permission.
 * - `pickVideo`: Video file picker with persistable permission.
 * - `pickMultipleFiles`: Multiple image/video selection.
 * - `pickFolder`: Directory selection (ACTION_OPEN_DOCUMENT_TREE) with persistable permission.
 * - `listFilesInFolder`: Scans child document files within a tree URI.
 */
class NativeFilePickerPlugin :
    FlutterPlugin,
    MethodChannel.MethodCallHandler,
    ActivityAware,
    PluginRegistry.ActivityResultListener {

    companion object {
        private const val TAG = "NativeFilePicker"
        private const val CHANNEL = "native_file_picker"
        private const val REQUEST_CODE_SINGLE = 9991
        private const val REQUEST_CODE_MULTIPLE = 9992
        private const val REQUEST_CODE_FOLDER = 9993
        private const val REQUEST_CODE_VIDEO = 9994
    }

    private lateinit var channel: MethodChannel
    private var appContext: Context? = null
    private var activity: Activity? = null
    private var pendingResult: MethodChannel.Result? = null

    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        appContext = binding.applicationContext
        channel = MethodChannel(binding.binaryMessenger, CHANNEL)
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        pendingResult?.success(null)
        pendingResult = null
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        val act = activity
        if (act == null) {
            result.error("NO_ACTIVITY", "No active Activity to present the file picker.", null)
            return
        }

        if (call.method == "listFilesInFolder") {
            val folderUriStr = call.argument<String>("folderUri")
            if (folderUriStr == null) {
                result.error("INVALID_ARG", "folderUri must not be null.", null)
                return
            }
            ioScope.launch {
                try {
                    val treeUri = Uri.parse(folderUriStr)
                    val dir = DocumentFile.fromTreeUri(act, treeUri)
                    val filesList = mutableListOf<Map<String, Any?>>()
                    dir?.listFiles()?.forEach { file ->
                        val mime = file.type ?: ""
                        val isVideo = mime.startsWith("video/")
                        val isImage = mime.startsWith("image/")
                        if (isImage || isVideo || mime.isNotEmpty()) {
                            filesList.add(
                                mapOf(
                                    "uri" to file.uri.toString(),
                                    "name" to (file.name ?: "unnamed"),
                                    "size" to file.length(),
                                    "mimeType" to mime,
                                    "isVideo" to isVideo,
                                    "lastModified" to file.lastModified()
                                )
                            )
                        }
                    }
                    withContext(Dispatchers.Main) {
                        result.success(filesList)
                    }
                } catch (e: Exception) {
                    withContext(Dispatchers.Main) {
                        result.error("LIST_FILES_ERROR", e.message, null)
                    }
                }
            }
            return
        }

        if (pendingResult != null) {
            result.error("ALREADY_ACTIVE", "A file pick operation is already in progress.", null)
            return
        }

        pendingResult = result

        when (call.method) {
            "pickFile" -> {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                }
                try {
                    act.startActivityForResult(intent, REQUEST_CODE_SINGLE)
                } catch (e: Exception) {
                    pendingResult?.error("LAUNCH_ERROR", e.message, null)
                    pendingResult = null
                }
            }
            "pickVideo" -> {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "video/*"
                }
                try {
                    act.startActivityForResult(intent, REQUEST_CODE_VIDEO)
                } catch (e: Exception) {
                    pendingResult?.error("LAUNCH_ERROR", e.message, null)
                    pendingResult = null
                }
            }
            "pickMultipleFiles", "pickMultipleMedia" -> {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
                    type = "*/*"
                    putExtra(Intent.EXTRA_MIME_TYPES, arrayOf("image/*", "video/*"))
                }
                try {
                    act.startActivityForResult(intent, REQUEST_CODE_MULTIPLE)
                } catch (e: Exception) {
                    pendingResult?.error("LAUNCH_ERROR", e.message, null)
                    pendingResult = null
                }
            }
            "pickFolder" -> {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT_TREE).apply {
                    addFlags(
                        Intent.FLAG_GRANT_READ_URI_PERMISSION or
                        Intent.FLAG_GRANT_WRITE_URI_PERMISSION or
                        Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION or
                        Intent.FLAG_GRANT_PREFIX_URI_PERMISSION
                    )
                }
                try {
                    act.startActivityForResult(intent, REQUEST_CODE_FOLDER)
                } catch (e: Exception) {
                    pendingResult?.error("LAUNCH_ERROR", e.message, null)
                    pendingResult = null
                }
            }
            else -> {
                pendingResult = null
                result.notImplemented()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode != REQUEST_CODE_SINGLE &&
            requestCode != REQUEST_CODE_MULTIPLE &&
            requestCode != REQUEST_CODE_FOLDER &&
            requestCode != REQUEST_CODE_VIDEO) {
            return false
        }

        val pending = pendingResult
        pendingResult = null

        if (resultCode == Activity.RESULT_OK) {
            val act = activity
            when (requestCode) {
                REQUEST_CODE_SINGLE, REQUEST_CODE_VIDEO -> {
                    val uri: Uri? = data?.data
                    if (uri != null) {
                        try {
                            act?.contentResolver?.takePersistableUriPermission(
                                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                            )
                        } catch (e: Exception) {
                            Log.w(TAG, "Could not take persistable permission: ${e.message}")
                        }
                        pending?.success(uri.toString())
                    } else {
                        pending?.success(null)
                    }
                }
                REQUEST_CODE_FOLDER -> {
                    val treeUri: Uri? = data?.data
                    if (treeUri != null) {
                        try {
                            val flags = Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                            act?.contentResolver?.takePersistableUriPermission(treeUri, flags)
                        } catch (e: Exception) {
                            Log.w(TAG, "Could not take persistable tree permission: ${e.message}")
                        }
                        pending?.success(treeUri.toString())
                    } else {
                        pending?.success(null)
                    }
                }
                REQUEST_CODE_MULTIPLE -> {
                    val uriList = mutableListOf<String>()
                    val clipData = data?.clipData
                    if (clipData != null) {
                        for (i in 0 until clipData.itemCount) {
                            val uri = clipData.getItemAt(i).uri
                            try {
                                act?.contentResolver?.takePersistableUriPermission(
                                    uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                                )
                            } catch (_: Exception) {}
                            uriList.add(uri.toString())
                        }
                    } else if (data?.data != null) {
                        val uri = data.data!!
                        try {
                            act?.contentResolver?.takePersistableUriPermission(
                                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
                            )
                        } catch (_: Exception) {}
                        uriList.add(uri.toString())
                    }
                    pending?.success(uriList)
                }
            }
        } else {
            pending?.success(null)
        }

        return true
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activity = binding.activity
        binding.addActivityResultListener(this)
    }

    override fun onDetachedFromActivityForConfigChanges() {
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