import 'package:flutter_test/flutter_test.dart';
import 'package:go_solving_app/models/board_state.dart';

void main() {
  group('BoardState', () {
    test('creates empty board with correct size', () {
      final board = BoardState(boardSize: 19);
      expect(board.boardSize, 19);
      expect(board.isEmpty, true);
      expect(board.blackCount, 0);
      expect(board.whiteCount, 0);
    });

    test('supports different board sizes', () {
      for (final size in [9, 13, 19]) {
        final board = BoardState(boardSize: size);
        expect(board.grid.length, size);
        expect(board.grid[0].length, size);
      }
    });

    test('setStone returns new BoardState', () {
      final board = BoardState(boardSize: 9);
      final newBoard = board.setStone(0, 0, StoneColor.black);

      expect(board.getStone(0, 0), StoneColor.empty);
      expect(newBoard.getStone(0, 0), StoneColor.black);
      expect(newBoard.blackCount, 1);
    });

    test('getStone throws on out of bounds', () {
      final board = BoardState(boardSize: 9);
      expect(() => board.getStone(-1, 0), throwsRangeError);
      expect(() => board.getStone(0, 9), throwsRangeError);
    });

    test('setStone throws on out of bounds', () {
      final board = BoardState(boardSize: 9);
      expect(() => board.setStone(9, 0, StoneColor.black), throwsRangeError);
    });

    test('default next player is black', () {
      final board = BoardState(boardSize: 19);
      expect(board.nextPlayer, StoneColor.black);
    });

    test('copyWithNextPlayer switches player', () {
      final board = BoardState(boardSize: 19);
      final switched = board.copyWithNextPlayer(StoneColor.white);
      expect(switched.nextPlayer, StoneColor.white);
    });

    test('counts stones correctly', () {
      var board = BoardState(boardSize: 9);
      board = board.setStone(0, 0, StoneColor.black);
      board = board.setStone(0, 1, StoneColor.black);
      board = board.setStone(1, 0, StoneColor.white);
      expect(board.blackCount, 2);
      expect(board.whiteCount, 1);
    });

    test('toFlatList returns correct length', () {
      final board = BoardState(boardSize: 9);
      expect(board.toFlatList().length, 81);

      final board19 = BoardState(boardSize: 19);
      expect(board19.toFlatList().length, 361);
    });

    test('toString produces readable output', () {
      var board = BoardState(boardSize: 3);
      board = board.setStone(0, 0, StoneColor.black);
      board = board.setStone(1, 1, StoneColor.white);
      final str = board.toString();
      expect(str, contains('X'));
      expect(str, contains('O'));
      expect(str, contains('.'));
    });

    test('default komi is 7.5', () {
      final board = BoardState(boardSize: 19);
      expect(board.komi, 7.5);
    });
  });

  group('StoneColor', () {
    test('opponent returns correct value', () {
      expect(StoneColor.black.opponent, StoneColor.white);
      expect(StoneColor.white.opponent, StoneColor.black);
      expect(StoneColor.empty.opponent, StoneColor.empty);
    });
  });

  group('BoardPosition', () {
    test('equality', () {
      expect(const BoardPosition(3, 4), const BoardPosition(3, 4));
      expect(const BoardPosition(3, 4) == const BoardPosition(3, 5), false);
    });

    test('hashCode is consistent', () {
      expect(
        const BoardPosition(3, 4).hashCode,
        const BoardPosition(3, 4).hashCode,
      );
    });
  });
}
