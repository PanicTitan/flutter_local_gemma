import 'package:flutter/material.dart';
import 'app_state.dart';
import 'screens/chat_screen.dart';
import 'screens/embedding_screen.dart';
import 'screens/smart_chat_screen.dart';
import 'screens/benchmark_screen.dart';
import 'screens/test_runner_screen.dart';

void main() {
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
    EmbeddingScreen(),
    SmartChatScreen(),
    BenchmarkScreen(),
    TestRunnerScreen(),   // ← new
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
          NavigationDestination(icon: Icon(Icons.hub_outlined),          selectedIcon: Icon(Icons.hub),           label: 'Embeddings'),
          NavigationDestination(icon: Icon(Icons.auto_awesome_outlined), selectedIcon: Icon(Icons.auto_awesome),  label: 'Smart Chat'),
          NavigationDestination(icon: Icon(Icons.speed_outlined),        selectedIcon: Icon(Icons.speed),         label: 'Benchmark'),
          NavigationDestination(icon: Icon(Icons.science_outlined),      selectedIcon: Icon(Icons.science),       label: 'Tests'),
        ],
      ),
    );
  }
}