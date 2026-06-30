import 'package:flutter/material.dart';

class SkillsScreen extends StatelessWidget {
  const SkillsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Skills')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: const [
          ListTile(title: Text('Calculator'), subtitle: Text('Evaluate mathematical expressions')),
          ListTile(title: Text('QR Code'), subtitle: Text('Generate QR codes')),
          ListTile(title: Text('Wikipedia'), subtitle: Text('Search Wikipedia')),
        ],
      ),
    );
  }
}
