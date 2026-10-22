import 'package:flutter/material.dart';
import 'app_state.dart';
import 'screens/chat_screen.dart';
import 'screens/models_screen.dart';
import 'screens/embedding_screen.dart';
import 'screens/gallery_search_screen.dart';
import 'screens/video_moment_screen.dart';
import 'screens/background_embedding_screen.dart';
import 'screens/benchmark_screen.dart';
import 'screens/test_runner_screen.dart';
import 'services/foreground_task_controller.dart';
import 'package:flutter_local_gemma/utils/gemma_debug.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  GemmaDebug.enabled = true;
  await GemmaDebug.setNativeDebug(true);

  // Starts/stops a "keep the app alive while busy" mechanism automatically
  // whenever GemmaState.instance.inferenceState goes busy/idle: an Android
  // foreground service on mobile, a silent looping <audio> element on web.
  // See lib/services/foreground_task_controller*.dart and the "Keep app
  // active while generating" toggle in the chat screen's settings drawer.
  await ForegroundTaskController.instance.init();

  runApp(const AIPlaygroundApp());
}

class AIPlaygroundApp extends StatelessWidget {
  const AIPlaygroundApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: ModelManager.instance,
      builder: (_, __) => MaterialApp(
        title: 'AI Playground',
        debugShowCheckedModeBanner: false,
        theme:     ThemeData(useMaterial3: true, fontFamily: 'system-ui', colorSchemeSeed: Colors.indigo, brightness: Brightness.light),
        darkTheme: ThemeData(useMaterial3: true, fontFamily: 'system-ui', colorSchemeSeed: Colors.indigo, brightness: Brightness.dark),
        themeMode: ThemeMode.system,
        home: const _MainShell(),
      ),
    );
  }
}

class _MainShell extends StatefulWidget {
  const _MainShell();
  @override
  State<_MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<_MainShell> {
  int _currentIndex = 0;
  final Set<int> _loadedTabs = {0};

  static final List<Widget Function()> _screenBuilders = [
    () => const ChatScreen(),
    () => const ModelsScreen(),
    () => const GallerySearchScreen(),
    () => const VideoMomentScreen(),
    () => const EmbeddingScreen(),
    () => const BackgroundEmbeddingScreen(),
    () => const BenchmarkScreen(),
    () => const TestRunnerScreen(),
  ];

  late final List<Widget?> _loadedScreens =
      List<Widget?>.filled(_screenBuilders.length, null);

  @override
  Widget build(BuildContext context) {
    _loadedTabs.add(_currentIndex);
    for (final index in _loadedTabs) {
      _loadedScreens[index] ??= _screenBuilders[index]();
    }

    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: List.generate(_loadedScreens.length, (i) {
          return _loadedScreens[i] ?? const SizedBox.shrink();
        }),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        labelBehavior: NavigationDestinationLabelBehavior.onlyShowSelected,
        destinations: const [
          NavigationDestination(icon: Icon(Icons.chat_bubble_outline),  selectedIcon: Icon(Icons.chat_bubble),   label: 'Chat'),
          NavigationDestination(icon: Icon(Icons.memory_outlined),      selectedIcon: Icon(Icons.memory),        label: 'Models'),
          NavigationDestination(icon: Icon(Icons.photo_library_outlined), selectedIcon: Icon(Icons.photo_library), label: 'Gallery'),
          NavigationDestination(icon: Icon(Icons.video_library_outlined), selectedIcon: Icon(Icons.video_library), label: 'Moments'),
          NavigationDestination(icon: Icon(Icons.hub_outlined),         selectedIcon: Icon(Icons.hub),           label: 'Embeds'),
          NavigationDestination(icon: Icon(Icons.sync_outlined),        selectedIcon: Icon(Icons.sync),          label: 'BG Tasks'),
          NavigationDestination(icon: Icon(Icons.speed_outlined),       selectedIcon: Icon(Icons.speed),         label: 'Bench'),
          NavigationDestination(icon: Icon(Icons.science_outlined),     selectedIcon: Icon(Icons.science),       label: 'Tests'),
        ],
      ),
    );
  }
}