import 'package:flutter_test/flutter_test.dart';
import 'package:go_solving_app/services/board_recognition.dart';
import 'package:go_solving_app/models/board_state.dart';

void main() {
  test('BoardRecognition - generateSampleBoard works', () {
    final service = BoardRecognition();
    final board = service.generateSampleBoard();

    expect(board.boardSize, 19);
    expect(board.grid.length, 19);
    expect(board.grid[0].length, 19);
    expect(board.grid[3][3], StoneColor.black);
  });

  // Note: We cannot test recognizeFromImage here because it requires
  // native OpenCV libraries which might not be loaded in the test environment
  // or require a real image path.
}
