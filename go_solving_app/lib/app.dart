import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'screens/camera_screen.dart';
import 'screens/analysis_screen.dart';

class GoSolvingApp extends MaterialApp {
  GoSolvingApp({super.key})
      : super(
          title: '圍棋解題',
          theme: ThemeData(
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.brown,
              brightness: Brightness.light,
            ),
            useMaterial3: true,
            appBarTheme: const AppBarTheme(
              centerTitle: true,
            ),
          ),
          darkTheme: ThemeData(
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.brown,
              brightness: Brightness.dark,
            ),
            useMaterial3: true,
            appBarTheme: const AppBarTheme(
              centerTitle: true,
            ),
          ),
          initialRoute: '/',
          routes: {
            '/': (context) => const HomeScreen(),
            '/camera': (context) => const CameraScreen(),
            '/analysis': (context) => const AnalysisScreen(),
          },
        );
}
