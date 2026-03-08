import 'package:flutter_test/flutter_test.dart';
import '../lib/models/board_state.dart';

void main() {
  test('playMove captures opponent stones and prevents immediate ko recapture', () {
    // 19x19 board
    var board = BoardState(boardSize: 19);

    // Setup a basic ko shape for black to capture
    // Black at (1,2), (2,1), (3,2)
    // White at (1,3), (2,4), (3,3)
    // Black plays at (2,3), capturing White at (2,2)
    // But White is at (2,2)
    board = board.setStone(1, 2, StoneColor.black);
    board = board.setStone(2, 1, StoneColor.black);
    board = board.setStone(3, 2, StoneColor.black);
    board = board.setStone(1, 3, StoneColor.white);
    board = board.setStone(2, 4, StoneColor.white);
    board = board.setStone(3, 3, StoneColor.white);
    board = board.setStone(2, 2, StoneColor.white);

    // Switch to black's turn
    board = board.copyWithNextPlayer(StoneColor.black);

    // Black plays at (2, 3), should capture white at (2, 2)
    board = board.playMove(2, 3);

    expect(board.getStone(2, 2), equals(StoneColor.empty));
    expect(board.getStone(2, 3), equals(StoneColor.black));

    // White tries to play at (2, 2) to recapture immediately, should throw ko error
    expect(() => board.playMove(2, 2), throwsA(isA<StateError>()));
  });
}
