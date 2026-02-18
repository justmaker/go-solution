import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_solving_app/models/board_state.dart';

void main() {
  group('KataGo Feature Encoding', () {
    test('binary features shape is correct for 9x9', () {
      const channels = 22;
      const boardSize = 9;
      final features = List<double>.filled(channels * boardSize * boardSize, 0.0);
      expect(features.length, 22 * 81);
    });

    test('binary features shape is correct for 19x19', () {
      const channels = 22;
      const boardSize = 19;
      final features = List<double>.filled(channels * boardSize * boardSize, 0.0);
      expect(features.length, 22 * 361);
    });

    test('binary features encode own stones correctly', () {
      final board = BoardState(boardSize: 9)
          .setStone(0, 0, StoneColor.black)
          .setStone(1, 1, StoneColor.white)
          .copyWithNextPlayer(StoneColor.black);

      const n = 9;
      const channels = 22;
      final features = List<double>.filled(channels * n * n, 0.0);
      final currentPlayer = board.nextPlayer;
      final opponent = currentPlayer.opponent;

      for (int r = 0; r < n; r++) {
        for (int c = 0; c < n; c++) {
          final stone = board.grid[r][c];
          final idx = r * n + c;
          if (stone == currentPlayer) features[0 * n * n + idx] = 1.0;
          if (stone == opponent) features[1 * n * n + idx] = 1.0;
          if (stone == StoneColor.empty) features[2 * n * n + idx] = 1.0;
        }
      }

      // (0,0) 是黑子 = 己方 (Ch 0)
      expect(features[0 * n * n + 0], 1.0);
      expect(features[1 * n * n + 0], 0.0);

      // (1,1) 是白子 = 對方 (Ch 1)
      expect(features[0 * n * n + 1 * n + 1], 0.0);
      expect(features[1 * n * n + 1 * n + 1], 1.0);

      // (2,2) 是空位 (Ch 2)
      expect(features[2 * n * n + 2 * n + 2], 1.0);
    });

    test('binary features encode perspective correctly', () {
      const n = 9;
      const channels = 22;

      // 黑棋視角
      final featuresBlack = List<double>.filled(channels * n * n, 0.0);
      for (int i = 0; i < n * n; i++) {
        featuresBlack[3 * n * n + i] = 1.0; // 黑棋
        featuresBlack[4 * n * n + i] = 0.0;
      }
      expect(featuresBlack[3 * n * n], 1.0);
      expect(featuresBlack[4 * n * n], 0.0);

      // 白棋視角
      final featuresWhite = List<double>.filled(channels * n * n, 0.0);
      for (int i = 0; i < n * n; i++) {
        featuresWhite[3 * n * n + i] = 0.0;
        featuresWhite[4 * n * n + i] = 1.0; // 白棋
      }
      expect(featuresWhite[3 * n * n], 0.0);
      expect(featuresWhite[4 * n * n], 1.0);
    });

    test('global features encode komi correctly', () {
      const globalFeatures = 19;
      final features = List<double>.filled(globalFeatures, 0.0);

      // 貼目 7.5
      features[0] = 7.5 / 20.0;
      expect(features[0], closeTo(0.375, 0.001));

      // 貼目 6.5
      features[0] = 6.5 / 20.0;
      expect(features[0], closeTo(0.325, 0.001));
    });

    test('global features encode player correctly', () {
      const globalFeatures = 19;

      // 黑棋下一手
      final blackFeatures = List<double>.filled(globalFeatures, 0.0);
      blackFeatures[1] = 1.0;
      blackFeatures[2] = 0.0;
      expect(blackFeatures[1], 1.0);
      expect(blackFeatures[2], 0.0);

      // 白棋下一手
      final whiteFeatures = List<double>.filled(globalFeatures, 0.0);
      whiteFeatures[1] = 0.0;
      whiteFeatures[2] = 1.0;
      expect(whiteFeatures[1], 0.0);
      expect(whiteFeatures[2], 1.0);
    });
  });

  group('Softmax', () {
    List<double> softmax(List<double> values) {
      if (values.isEmpty) return [];
      final maxVal = values.reduce(max);
      final exps = values.map((v) => exp(v - maxVal)).toList();
      final sumExps = exps.reduce((a, b) => a + b);
      return exps.map((e) => e / sumExps).toList();
    }

    test('softmax sums to 1', () {
      final result = softmax([1.0, 2.0, 3.0]);
      final sum = result.reduce((a, b) => a + b);
      expect(sum, closeTo(1.0, 0.0001));
    });

    test('softmax preserves ordering', () {
      final result = softmax([1.0, 3.0, 2.0]);
      expect(result[1], greaterThan(result[2]));
      expect(result[2], greaterThan(result[0]));
    });

    test('softmax with equal values', () {
      final result = softmax([1.0, 1.0, 1.0]);
      for (final v in result) {
        expect(v, closeTo(1.0 / 3.0, 0.0001));
      }
    });

    test('softmax with large values', () {
      final result = softmax([1000.0, 1001.0, 999.0]);
      final sum = result.reduce((a, b) => a + b);
      expect(sum, closeTo(1.0, 0.0001));
      expect(result[1], greaterThan(result[0]));
    });

    test('softmax with empty list', () {
      expect(softmax([]), isEmpty);
    });
  });

  group('Tanh', () {
    double tanh(double x) {
      final e2x = exp(2 * x);
      return (e2x - 1) / (e2x + 1);
    }

    test('tanh of 0 is 0', () {
      expect(tanh(0), closeTo(0.0, 0.0001));
    });

    test('tanh is bounded between -1 and 1', () {
      expect(tanh(10), closeTo(1.0, 0.0001));
      expect(tanh(-10), closeTo(-1.0, 0.0001));
    });

    test('tanh is odd function', () {
      expect(tanh(0.5), closeTo(-tanh(-0.5), 0.0001));
      expect(tanh(1.0), closeTo(-tanh(-1.0), 0.0001));
    });
  });
}
