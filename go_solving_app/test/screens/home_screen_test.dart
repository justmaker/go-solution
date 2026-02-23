import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_solving_app/screens/home_screen.dart';

class MockNavigatorObserver extends NavigatorObserver {
  final List<Route<dynamic>> pushedRoutes = [];

  @override
  void didPush(Route<dynamic> route, Route<dynamic>? previousRoute) {
    pushedRoutes.add(route);
    super.didPush(route, previousRoute);
  }
}

void main() {
  testWidgets('Verify navigation to demo AnalysisScreen', (WidgetTester tester) async {
    final mockObserver = MockNavigatorObserver();

    await tester.pumpWidget(
      MaterialApp(
        initialRoute: '/',
        routes: {
          '/': (context) => const HomeScreen(),
          '/analysis': (context) => const Scaffold(body: Text('Analysis Screen')),
        },
        navigatorObservers: [mockObserver],
      ),
    );

    // Verify we are on the Home Screen
    expect(find.text('圍棋解題'), findsWidgets);

    // Find the demo button by its label
    final demoButton = find.text('範例棋盤（測試用）');
    expect(demoButton, findsOneWidget);

    // Tap the button
    await tester.tap(demoButton);

    // Pump the widget to trigger the navigation
    await tester.pumpAndSettle();

    // Verify navigation occurred
    // pushedRoutes[0] is the initial '/' route
    // pushedRoutes[1] should be the '/analysis' route
    expect(mockObserver.pushedRoutes.length, 2);

    final Route<dynamic> route = mockObserver.pushedRoutes.last;
    expect(route.settings.name, '/analysis');
    expect(route.settings.arguments, 'demo');

    // Verify we reached the destination
    expect(find.text('Analysis Screen'), findsOneWidget);
  });
}
