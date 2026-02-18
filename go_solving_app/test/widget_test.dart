import 'package:flutter_test/flutter_test.dart';
import 'package:go_solving_app/app.dart';

void main() {
  testWidgets('App renders home screen', (WidgetTester tester) async {
    await tester.pumpWidget(GoSolvingApp());

    expect(find.text('圍棋解題'), findsWidgets);
    expect(find.text('拍照辨識'), findsOneWidget);
    expect(find.text('從相簿選取'), findsOneWidget);
  });
}
