import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_solving_app/models/board_state.dart';
import 'package:go_solving_app/models/analysis_result.dart';
import 'package:go_solving_app/widgets/board_painter.dart';
import 'package:go_solving_app/widgets/stone_widget.dart';
import 'package:go_solving_app/widgets/move_suggestion.dart';

void main() {
  group('BoardPainterWidget', () {
    testWidgets('renders without error for empty 9x9 board', (tester) async {
      final board = BoardState(boardSize: 9);
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: BoardPainterWidget(boardState: board),
          ),
        ),
      );

      expect(find.byType(BoardPainterWidget), findsOneWidget);
      expect(find.byType(CustomPaint), findsWidgets);
    });

    testWidgets('renders without error for 19x19 board', (tester) async {
      final board = BoardState(boardSize: 19);
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: BoardPainterWidget(boardState: board),
          ),
        ),
      );

      expect(find.byType(BoardPainterWidget), findsOneWidget);
    });

    testWidgets('renders with stones on board', (tester) async {
      var board = BoardState(boardSize: 9);
      board = board.setStone(4, 4, StoneColor.black);
      board = board.setStone(3, 3, StoneColor.white);

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: BoardPainterWidget(boardState: board),
          ),
        ),
      );

      expect(find.byType(BoardPainterWidget), findsOneWidget);
    });

    testWidgets('renders with analysis result', (tester) async {
      var board = BoardState(boardSize: 9);
      board = board.setStone(4, 4, StoneColor.black);

      const result = AnalysisResult(
        topMoves: [
          MoveSuggestion(
            position: BoardPosition(3, 3),
            probability: 0.5,
            rank: 1,
          ),
          MoveSuggestion(
            position: BoardPosition(5, 5),
            probability: 0.3,
            rank: 2,
          ),
        ],
        winrate: [0.55, 0.45, 0.0],
        boardSize: 9,
        nextPlayer: StoneColor.black,
      );

      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: BoardPainterWidget(
              boardState: board,
              analysisResult: result,
            ),
          ),
        ),
      );

      expect(find.byType(BoardPainterWidget), findsOneWidget);
    });
  });

  group('StoneWidget', () {
    testWidgets('renders black stone', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: StoneWidget(color: StoneColor.black, size: 48),
          ),
        ),
      );

      expect(find.byType(StoneWidget), findsOneWidget);
    });

    testWidgets('renders white stone', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: StoneWidget(color: StoneColor.white, size: 48),
          ),
        ),
      );

      expect(find.byType(StoneWidget), findsOneWidget);
    });

    testWidgets('renders empty as SizedBox', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: StoneWidget(color: StoneColor.empty, size: 48),
          ),
        ),
      );

      expect(find.byType(SizedBox), findsWidgets);
    });

    testWidgets('renders with label', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: StoneWidget(
              color: StoneColor.black,
              size: 48,
              label: '1',
            ),
          ),
        ),
      );

      expect(find.text('1'), findsOneWidget);
    });
  });

  group('MoveSuggestionWidget', () {
    testWidgets('renders top moves', (tester) async {
      const result = AnalysisResult(
        topMoves: [
          MoveSuggestion(
            position: BoardPosition(3, 3),
            probability: 0.5,
            rank: 1,
          ),
          MoveSuggestion(
            position: BoardPosition(5, 5),
            probability: 0.3,
            rank: 2,
          ),
          MoveSuggestion(
            position: BoardPosition(2, 6),
            probability: 0.1,
            rank: 3,
          ),
        ],
        winrate: [0.55, 0.45, 0.0],
        boardSize: 9,
        nextPlayer: StoneColor.black,
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: MoveSuggestionWidget(result: result),
          ),
        ),
      );

      expect(find.text('最佳著手建議'), findsOneWidget);
      // 座標 D6, F4, G7 和機率百分比
      expect(find.text('D6'), findsWidgets);
      expect(find.text('50.0%'), findsWidgets);
      expect(find.text('30.0%'), findsWidgets);
      expect(find.text('10.0%'), findsWidgets);
    });

    testWidgets('renders empty state when no moves', (tester) async {
      const result = AnalysisResult(
        topMoves: [],
        winrate: [0.5, 0.5, 0.0],
        boardSize: 9,
        nextPlayer: StoneColor.black,
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: MoveSuggestionWidget(result: result),
          ),
        ),
      );

      expect(find.text('無建議著手'), findsOneWidget);
    });
  });
}
