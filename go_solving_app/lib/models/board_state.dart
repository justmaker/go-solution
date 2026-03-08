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
  final List<List<StoneColor>>? initialGrid;
  final List<List<StoneColor>>? previousGrid;
  final StoneColor nextPlayer;
  final List<BoardPosition> moveHistory;
  final double komi;

  BoardState({
    required this.boardSize,
    List<List<StoneColor>>? grid,
    this.initialGrid,
    this.previousGrid,
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
      initialGrid: initialGrid,
      previousGrid: previousGrid,
      nextPlayer: nextPlayer,
      moveHistory: List.from(moveHistory),
      komi: komi,
    );
  }

  // Helpers to get adjacent positions
  List<BoardPosition> _getAdjacent(int row, int col) {
    final adj = <BoardPosition>[];
    if (row > 0) adj.add(BoardPosition(row - 1, col));
    if (row < boardSize - 1) adj.add(BoardPosition(row + 1, col));
    if (col > 0) adj.add(BoardPosition(row, col - 1));
    if (col < boardSize - 1) adj.add(BoardPosition(row, col + 1));
    return adj;
  }

  // Helper to find a group of stones of the same color
  Set<BoardPosition> _getGroup(int startRow, int startCol, List<List<StoneColor>> currentGrid) {
    final color = currentGrid[startRow][startCol];
    final group = <BoardPosition>{};
    final queue = [BoardPosition(startRow, startCol)];

    while (queue.isNotEmpty) {
      final pos = queue.removeLast();
      if (!group.contains(pos)) {
        group.add(pos);
        for (final adj in _getAdjacent(pos.row, pos.col)) {
          if (currentGrid[adj.row][adj.col] == color && !group.contains(adj)) {
            queue.add(adj);
          }
        }
      }
    }
    return group;
  }

  // Helper to get liberties of a group
  int _getLiberties(Set<BoardPosition> group, List<List<StoneColor>> currentGrid) {
    final liberties = <BoardPosition>{};
    for (final pos in group) {
      for (final adj in _getAdjacent(pos.row, pos.col)) {
        if (currentGrid[adj.row][adj.col] == StoneColor.empty) {
          liberties.add(adj);
        }
      }
    }
    return liberties.length;
  }

  bool _gridsEqual(List<List<StoneColor>> grid1, List<List<StoneColor>> grid2) {
    for (int r = 0; r < boardSize; r++) {
      for (int c = 0; c < boardSize; c++) {
        if (grid1[r][c] != grid2[r][c]) {
          return false;
        }
      }
    }
    return true;
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

    // Evaluate captures
    bool capturedAny = false;
    final opponent = nextPlayer.opponent;
    for (final adj in _getAdjacent(row, col)) {
      if (newGrid[adj.row][adj.col] == opponent) {
        final group = _getGroup(adj.row, adj.col, newGrid);
        if (_getLiberties(group, newGrid) == 0) {
          capturedAny = true;
          for (final pos in group) {
            newGrid[pos.row][pos.col] = StoneColor.empty;
          }
        }
      }
    }

    // Suicide prevention
    if (!capturedAny) {
      final selfGroup = _getGroup(row, col, newGrid);
      if (_getLiberties(selfGroup, newGrid) == 0) {
        throw StateError('Suicide move is not allowed');
      }
    }

    // Ko rule
    if (previousGrid != null && _gridsEqual(newGrid, previousGrid!)) {
      throw StateError('Ko rule violation');
    }

    final newHistory = List<BoardPosition>.from(moveHistory)
      ..add(BoardPosition(row, col));

    // If initialGrid isn't set yet (i.e. this is the first played move),
    // save the current grid as the starting point.
    final initGrid = initialGrid ?? grid;

    return BoardState(
      boardSize: boardSize,
      grid: newGrid,
      initialGrid: initGrid,
      previousGrid: grid, // Current grid becomes previous grid
      nextPlayer: opponent,
      moveHistory: newHistory,
      komi: komi,
    );
  }

  /// 複製棋盤並切換下一手玩家
  BoardState copyWithNextPlayer(StoneColor player) {
    return BoardState(
      boardSize: boardSize,
      grid: List.generate(boardSize, (r) => List<StoneColor>.from(grid[r])),
      initialGrid: initialGrid,
      previousGrid: previousGrid,
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
