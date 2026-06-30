import 'package:flutter/material.dart';
import 'app_state.dart';
import 'screens/chat_screen.dart';
import 'screens/models_screen.dart';
import 'screens/embedding_screen.dart';
import 'screens/benchmark_screen.dart';
import 'screens/test_runner_screen.dart';
import 'package:flutter_local_gemma/utils/gemma_debug.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  GemmaDebug.enabled = true;
  await GemmaDebug.setNativeDebug(true);
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

  static const _screens = [
    ChatScreen(),
    ModelsScreen(),
    EmbeddingScreen(),
    BenchmarkScreen(),
    TestRunnerScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(index: _currentIndex, children: _screens),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _currentIndex,
        onDestinationSelected: (i) => setState(() => _currentIndex = i),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.chat_bubble_outline),  selectedIcon: Icon(Icons.chat_bubble),   label: 'Chat'),
          NavigationDestination(icon: Icon(Icons.memory_outlined),      selectedIcon: Icon(Icons.memory),        label: 'Models'),
          NavigationDestination(icon: Icon(Icons.hub_outlined),         selectedIcon: Icon(Icons.hub),           label: 'Embeds'),
          NavigationDestination(icon: Icon(Icons.speed_outlined),       selectedIcon: Icon(Icons.speed),         label: 'Bench'),
          NavigationDestination(icon: Icon(Icons.science_outlined),     selectedIcon: Icon(Icons.science),       label: 'Tests'),
        ],
      ),
    );
  }
}