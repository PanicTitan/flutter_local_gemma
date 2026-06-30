package com.example.flutter_local_gemma

import android.annotation.SuppressLint
import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient
import kotlinx.coroutines.*
import org.json.JSONObject
import java.util.concurrent.ConcurrentHashMap

private const val TAG = "JsSkillExecutor"
private const val ASSET_BASE_URL = "https://appassets.androidplatform.net/assets/skills"

class JsSkillExecutor(private val context: Context) {

    private val mainHandler = Handler(Looper.getMainLooper())
    private val pool = ConcurrentHashMap<String, WebView>()
    private val interfacePool = ConcurrentHashMap<String, SkillJavascriptInterface>()

    private class SkillJavascriptInterface {
        var onResultListener: ((String) -> Unit)? = null

        @JavascriptInterface
        fun onResultReady(result: String) {
            onResultListener?.invoke(result)
        }
    }

    fun prewarm(skillName: String, scriptHtml: String) {
        if (pool.containsKey(skillName)) return
        mainHandler.post {
            val wv = buildWebView()
            val jsInterface = SkillJavascriptInterface()
            pool[skillName] = wv
            interfacePool[skillName] = jsInterface
            wv.addJavascriptInterface(jsInterface, "AiEdgeGallery")
            wv.loadDataWithBaseURL(
                "$ASSET_BASE_URL/$skillName/scripts/",
                scriptHtml, "text/html", "UTF-8", null
            )
            Log.d(TAG, "Pre-warming '$skillName'")
        }
    }

    suspend fun execute(
        skillName:  String,
        scriptName: String = "index.html",
        scriptHtml: String,
        argsJson:   String,
        secret:     String = "",
        timeoutMs:  Long   = 30_000L,
    ): String = withTimeout(timeoutMs) {
        suspendCancellableCoroutine { cont ->
            mainHandler.post {
                var isNew = false
                val wv = pool.getOrPut(skillName) {
                    isNew = true
                    buildWebView()
                }
                val jsInterface = interfacePool.getOrPut(skillName) {
                    SkillJavascriptInterface().also {
                        wv.addJavascriptInterface(it, "AiEdgeGallery")
                    }
                }

                if (isNew) {
                    wv.loadDataWithBaseURL(
                        "$ASSET_BASE_URL/$skillName/scripts/",
                        scriptHtml, "text/html", "UTF-8", null
                    )
                }

                jsInterface.onResultListener = { result ->
                    if (cont.isActive) cont.resume(result) {}
                }

                val safeData   = JSONObject.quote(argsJson)
                val safeSecret = JSONObject.quote(secret)
                val script = """
                    (async function() {
                        var startTs = Date.now();
                        while(true) {
                            if (typeof ai_edge_gallery_get_result === 'function') { break; }
                            await new Promise(resolve => { setTimeout(resolve, 100) });
                            if (Date.now() - startTs > 10000) { break; }
                        }
                        try {
                            var rawData = $safeData;
                            
                            // Standard Edge Gallery protocol expects a JSON string
                            var result = await ai_edge_gallery_get_result(rawData, $safeSecret);
                            
                            if (typeof result === 'object' && result !== null) {
                                AiEdgeGallery.onResultReady(JSON.stringify(result));
                            } else {
                                AiEdgeGallery.onResultReady(String(result));
                            }
                        } catch(e) {
                            AiEdgeGallery.onResultReady(JSON.stringify({
                                error: e.message || String(e), result: null
                            }));
                        }
                    })()
                """.trimIndent()

                if (isNew) {
                    wv.webViewClient = object : WebViewClient() {
                        override fun onPageFinished(view: WebView?, url: String?) {
                            wv.webViewClient = WebViewClient() // clear to prevent memory leaks
                            wv.evaluateJavascript(script, null)
                        }
                    }
                } else {
                    wv.evaluateJavascript(script, null)
                }
            }
        }
    }

    fun dispose() {
        mainHandler.post {
            pool.values.forEach {
                it.removeJavascriptInterface("AiEdgeGallery")
                it.destroy()
            }
            pool.clear()
            interfacePool.clear()
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun buildWebView(): WebView = WebView(context).apply {
        settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess   = false
        }
    }
}