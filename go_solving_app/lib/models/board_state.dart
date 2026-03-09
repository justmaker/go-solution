/// 棋盤上每個交叉點的狀態
enum StoneColor {
  empty,
  black,
  white;

  StoneColor get opponent {
    switch (this) {
      case StoneColor.black:
        return StoneColor.white;
      case StoneColor.white:
        return StoneColor.black;
      case StoneColor.empty:
        return StoneColor.empty;
    }
  }
}

/// 棋盤座標
class BoardPosition {
  final int row;
  final int col;

  const BoardPosition(this.row, this.col);

  @override
  bool operator ==(Object other) =>
      other is BoardPosition && other.row == row && other.col == col;

  @override
  int get hashCode => row * 31 + col;

  @override
  String toString() => '($row, $col)';
}

/// 棋盤狀態資料模型，支援任意大小（9x9, 13x13, 19x19）
class BoardState {
  final int boardSize;
  final List<List<StoneColor>> grid;
  final StoneColor nextPlayer;
  final List<BoardPosition> moveHistory;
  final double komi;

  BoardState({
    required this.boardSize,
    List<List<StoneColor>>? grid,
    this.nextPlayer = StoneColor.black,
    List<BoardPosition>? moveHistory,
    this.komi = 7.5,
  })  : grid = grid ??
            List.generate(
              boardSize,
              (_) => List.filled(boardSize, StoneColor.empty),
            ),
        moveHistory = moveHistory ?? [];

  /// 取得指定位置的棋子顏色
  StoneColor getStone(int row, int col) {
    if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
      throw RangeError('Position ($row, $col) out of bounds for $boardSize x $boardSize board');
    }
    return grid[row][col];
  }

  /// 設定指定位置的棋子，回傳新的 BoardState（immutable）
  BoardState setStone(int row, int col, StoneColor color) {
    if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
      throw RangeError('Position ($row, $col) out of bounds for $boardSize x $boardSize board');
    }
    final newGrid = List.generate(
      boardSize,
      (r) => List<StoneColor>.from(grid[r]),
    );
    newGrid[row][col] = color;
    return BoardState(
      boardSize: boardSize,
      grid: newGrid,
      nextPlayer: nextPlayer,
      moveHistory: List.from(moveHistory),
      komi: komi,
    );
  }

  /// 在指定位置下子，回傳新的 BoardState（immutable）
  /// 此方法會更新 moveHistory 並切換 nextPlayer
  BoardState playMove(int row, int col) {
    if (row < 0 || row >= boardSize || col < 0 || col >= boardSize) {
      throw RangeError('Position ($row, $col) out of bounds for $boardSize x $boardSize board');
    }
    if (grid[row][col] != StoneColor.empty) {
      throw StateError('Position ($row, $col) is not empty');
    }

    final newGrid = List.generate(
      boardSize,
      (r) => List<StoneColor>.from(grid[r]),
    );
    newGrid[row][col] = nextPlayer;

    // 檢查鄰近的對手棋子是否被提吃
    final opponentColor = nextPlayer.opponent;
    final adjacentPositions = [
      (row - 1, col),
      (row + 1, col),
      (row, col - 1),
      (row, col + 1),
    ];

    int capturedStones = 0;
    for (final pos in adjacentPositions) {
      final r = pos.$1;
      final c = pos.$2;
      if (r >= 0 && r < boardSize && c >= 0 && c < boardSize) {
        if (newGrid[r][c] == opponentColor) {
          final group = _getGroup(newGrid, r, c);
          if (_getLiberties(newGrid, group) == 0) {
            for (final stonePos in group) {
              newGrid[stonePos.row][stonePos.col] = StoneColor.empty;
              capturedStones++;
            }
          }
        }
      }
    }

    // 檢查自殺步 (如果沒有提吃對手，且自己沒有氣)
    if (capturedStones == 0) {
      final group = _getGroup(newGrid, row, col);
      if (_getLiberties(newGrid, group) == 0) {
        throw StateError('Position ($row, $col) is a suicide move');
      }
    }

    final newHistory = List<BoardPosition>.from(moveHistory)
      ..add(BoardPosition(row, col));

    return BoardState(
      boardSize: boardSize,
      grid: newGrid,
      nextPlayer: nextPlayer.opponent,
      moveHistory: newHistory,
      komi: komi,
    );
  }

  Set<BoardPosition> _getGroup(List<List<StoneColor>> currentGrid, int startRow, int startCol) {
    final color = currentGrid[startRow][startCol];
    final group = <BoardPosition>{};
    final queue = <BoardPosition>[BoardPosition(startRow, startCol)];

    // Using a simple list as a queue is fine here since max board size is small (19x19),
    // but we can optimize it slightly by tracking the read index instead of removeAt(0).
    int readIndex = 0;

    while (readIndex < queue.length) {
      final pos = queue[readIndex++];
      if (group.contains(pos)) continue;

      group.add(pos);

      final adjacentPositions = [
        (pos.row - 1, pos.col),
        (pos.row + 1, pos.col),
        (pos.row, pos.col - 1),
        (pos.row, pos.col + 1),
      ];

      for (final adj in adjacentPositions) {
        final r = adj.$1;
        final c = adj.$2;
        if (r >= 0 && r < boardSize && c >= 0 && c < boardSize) {
          if (currentGrid[r][c] == color) {
            queue.add(BoardPosition(r, c));
          }
        }
      }
    }
    return group;
  }

  int _getLiberties(List<List<StoneColor>> currentGrid, Set<BoardPosition> group) {
    final liberties = <BoardPosition>{};

    for (final pos in group) {
      final adjacentPositions = [
        (pos.row - 1, pos.col),
        (pos.row + 1, pos.col),
        (pos.row, pos.col - 1),
        (pos.row, pos.col + 1),
      ];

      for (final adj in adjacentPositions) {
        final r = adj.$1;
        final c = adj.$2;
        if (r >= 0 && r < boardSize && c >= 0 && c < boardSize) {
          if (currentGrid[r][c] == StoneColor.empty) {
            liberties.add(BoardPosition(r, c));
          }
        }
      }
    }

    return liberties.length;
  }

  /// 複製棋盤並切換下一手玩家
  BoardState copyWithNextPlayer(StoneColor player) {
    return BoardState(
      boardSize: boardSize,
      grid: List.generate(boardSize, (r) => List<StoneColor>.from(grid[r])),
      nextPlayer: player,
      moveHistory: List.from(moveHistory),
      komi: komi,
    );
  }

  /// 計算棋盤上黑子數量
  int get blackCount {
    int count = 0;
    for (final row in grid) {
      for (final stone in row) {
        if (stone == StoneColor.black) count++;
      }
    }
    return count;
  }

  /// 計算棋盤上白子數量
  int get whiteCount {
    int count = 0;
    for (final row in grid) {
      for (final stone in row) {
        if (stone == StoneColor.white) count++;
      }
    }
    return count;
  }

  /// 檢查棋盤是否為空
  bool get isEmpty => blackCount == 0 && whiteCount == 0;

  /// 將棋盤轉換為 flat list（row-major order）
  List<StoneColor> toFlatList() {
    return grid.expand((row) => row).toList();
  }

  @override
  String toString() {
    final sb = StringBuffer();
    sb.writeln('BoardState ${boardSize}x$boardSize, next: $nextPlayer');
    for (int r = 0; r < boardSize; r++) {
      for (int c = 0; c < boardSize; c++) {
        switch (grid[r][c]) {
          case StoneColor.empty:
            sb.write('.');
          case StoneColor.black:
            sb.write('X');
          case StoneColor.white:
            sb.write('O');
        }
      }
      sb.writeln();
    }
    return sb.toString();
  }
}
