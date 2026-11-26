import 'dart:io';
import 'package:path_provider/path_provider.dart';
import '../models/board_state.dart';

/// SGF（Smart Game Format）匯出服務
///
/// 匯出策略：使用單一節點 + AB/AW setup properties 記錄「目前盤面」上
/// 每一顆棋子的位置，而不是重播 moveHistory。
/// 這樣可以保證：不論棋子是怎麼下上去的（包含提吃、辨識結果、手動調整等），
/// 匯出當下的 SGF 盤面一定和畫面上看到的盤面一致，下棋順序完全不重要。
class SgfService {
  /// 將棋盤狀態轉換為 SGF 字串
  String boardStateToSgf(BoardState board, {String? comment}) {
    final size = board.boardSize;
    final blackPoints = <String>[];
    final whitePoints = <String>[];

    for (int r = 0; r < size; r++) {
      for (int c = 0; c < size; c++) {
        final stone = board.grid[r][c];
        if (stone == StoneColor.empty) continue;
        final point = _toSgfCoord(r, c);
        if (stone == StoneColor.black) {
          blackPoints.add(point);
        } else {
          whitePoints.add(point);
        }
      }
    }

    final sb = StringBuffer();
    sb.write('(;FF[4]GM[1]CA[UTF-8]AP[go_solving_app]');
    sb.write('SZ[$size]');
    sb.write('KM[${board.komi}]');
    sb.write('PL[${board.nextPlayer == StoneColor.black ? 'B' : 'W'}]');

    if (blackPoints.isNotEmpty) {
      sb.write('AB${blackPoints.map((p) => '[$p]').join()}');
    }
    if (whitePoints.isNotEmpty) {
      sb.write('AW${whitePoints.map((p) => '[$p]').join()}');
    }
    if (comment != null && comment.isNotEmpty) {
      sb.write('C[${_escapeSgfText(comment)}]');
    }
    sb.write(')');
    return sb.toString();
  }

  /// 將目前棋盤狀態寫入 SGF 檔案，回傳檔案路徑
  Future<File> exportToFile(BoardState board, {String? comment}) async {
    final sgfContent = boardStateToSgf(board, comment: comment);
    final dir = await getApplicationDocumentsDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final file = File('${dir.path}/go_solving_$timestamp.sgf');
    return file.writeAsString(sgfContent);
  }

  /// (row, col) -> SGF 座標（兩個小寫字母，a-z 對應 0-25）
  String _toSgfCoord(int row, int col) {
    final colChar = String.fromCharCode('a'.codeUnitAt(0) + col);
    final rowChar = String.fromCharCode('a'.codeUnitAt(0) + row);
    return '$colChar$rowChar';
  }

  String _escapeSgfText(String text) {
    return text.replaceAll('\\', '\\\\').replaceAll(']', '\\]');
  }
}
