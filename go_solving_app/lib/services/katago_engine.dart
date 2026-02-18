import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:path/path.dart' as p;
import '../models/board_state.dart';
import '../models/analysis_result.dart';

/// KataGo ONNX 推論引擎
class KataGoEngine {
  static const String _modelAssetPath = 'assets/models/katago_b6c64.onnx';
  static const int _binaryChannels = 22;
  static const int _globalFeatures = 19;

  final OnnxRuntime _onnxRuntime = OnnxRuntime();
  OnnxRuntimeSession? _session;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  /// 初始化 ONNX Runtime 並載入模型
  Future<void> initialize() async {
    if (_isInitialized) return;

    // 從 assets 複製模型到臨時目錄
    final modelPath = await _copyModelToTemp();

    // 建立 ONNX Runtime Session
    _session = await _onnxRuntime.createSession(modelPath);
    _isInitialized = true;
  }

  /// 複製模型檔案到臨時目錄
  Future<String> _copyModelToTemp() async {
    final tempDir = await getTemporaryDirectory();
    final modelFile = File(p.join(tempDir.path, 'katago_b6c64.onnx'));

    if (!await modelFile.exists()) {
      final data = await rootBundle.load(_modelAssetPath);
      await modelFile.writeAsBytes(data.buffer.asUint8List());
    }

    return modelFile.path;
  }

  /// 分析棋盤狀態，回傳分析結果
  Future<AnalysisResult> analyze(BoardState board) async {
    if (!_isInitialized || _session == null) {
      throw StateError('KataGoEngine 尚未初始化，請先呼叫 initialize()');
    }

    final n = board.boardSize;

    // 編碼輸入特徵
    final binaryInput = _encodeBinaryFeatures(board);
    final globalInput = _encodeGlobalFeatures(board);

    // 建立輸入張量
    final inputs = <String, OnnxRuntimeTensor>{
      'input_binary': OnnxRuntimeTensor.fromList(
        binaryInput,
        [1, _binaryChannels, n, n],
      ),
      'input_global': OnnxRuntimeTensor.fromList(
        globalInput,
        [1, _globalFeatures],
      ),
    };

    // 執行推論
    final outputs = await _session!.run(inputs);

    // 解析輸出
    return _parseOutputs(outputs, board);
  }

  /// 編碼 22 個二進位空間特徵平面
  /// Shape: [1, 22, boardSize, boardSize]
  List<double> _encodeBinaryFeatures(BoardState board) {
    final n = board.boardSize;
    final features = List<double>.filled(_binaryChannels * n * n, 0.0);

    // 當前玩家和對手
    final currentPlayer = board.nextPlayer;
    final opponent = currentPlayer.opponent;

    for (int r = 0; r < n; r++) {
      for (int c = 0; c < n; c++) {
        final stone = board.grid[r][c];
        final idx = r * n + c;

        // Ch 0: 己方棋子
        if (stone == currentPlayer) {
          features[0 * n * n + idx] = 1.0;
        }
        // Ch 1: 對方棋子
        if (stone == opponent) {
          features[1 * n * n + idx] = 1.0;
        }
        // Ch 2: 空位
        if (stone == StoneColor.empty) {
          features[2 * n * n + idx] = 1.0;
        }
      }
    }

    // Ch 3-4: 視角指示（全 1 表示當前玩家為黑）
    final isBlack = currentPlayer == StoneColor.black;
    for (int i = 0; i < n * n; i++) {
      features[3 * n * n + i] = isBlack ? 1.0 : 0.0;
      features[4 * n * n + i] = isBlack ? 0.0 : 1.0;
    }

    // Ch 5-9: 過去 1-5 手位置（從 moveHistory 取得）
    for (int h = 0; h < 5 && h < board.moveHistory.length; h++) {
      final move = board.moveHistory[board.moveHistory.length - 1 - h];
      final idx = move.row * n + move.col;
      features[(5 + h) * n * n + idx] = 1.0;
    }

    // Ch 10-21: 其餘特徵設為 0（梯子、活棋、規則等）
    // 簡化處理：這些特徵在基本分析中影響較小

    return features;
  }

  /// 編碼 19 個全域特徵
  /// Shape: [1, 19]
  List<double> _encodeGlobalFeatures(BoardState board) {
    final features = List<double>.filled(_globalFeatures, 0.0);

    // Feature 0: 貼目（正規化）
    features[0] = board.komi / 20.0;

    // Feature 1-2: 當前玩家指示
    features[1] = board.nextPlayer == StoneColor.black ? 1.0 : 0.0;
    features[2] = board.nextPlayer == StoneColor.white ? 1.0 : 0.0;

    // Feature 3: 無劫禁（預設無劫）
    features[3] = 0.0;

    // Feature 4: 使用中國規則
    features[4] = 1.0;

    // Feature 5-18: 其餘規則與狀態特徵設為 0
    // 簡化處理

    return features;
  }

  /// 解析模型輸出
  AnalysisResult _parseOutputs(
    Map<String, OnnxRuntimeTensor> outputs,
    BoardState board,
  ) {
    final n = board.boardSize;

    // 解析 policy（著手機率）
    final policyTensor = outputs['output_policy'];
    final policyData = policyTensor?.data as List<double>? ?? [];

    // Softmax
    final policy = _softmax(policyData);

    // 取 top-N 最佳著手
    final moves = <MoveSuggestion>[];
    final indexed = List.generate(
      min(policy.length, n * n),
      (i) => MapEntry(i, policy[i]),
    );
    indexed.sort((a, b) => b.value.compareTo(a.value));

    for (int rank = 0; rank < min(5, indexed.length); rank++) {
      final idx = indexed[rank].key;
      final row = idx ~/ n;
      final col = idx % n;
      moves.add(MoveSuggestion(
        position: BoardPosition(row, col),
        probability: indexed[rank].value,
        rank: rank + 1,
      ));
    }

    // 解析 value（勝率）
    final valueTensor = outputs['output_value'];
    final valueData = valueTensor?.data as List<double>? ?? [0.5, 0.5, 0.0];
    final winrate = _softmax(valueData.take(3).toList());

    // 解析 ownership（地域歸屬）
    List<List<double>>? ownership;
    final ownershipTensor = outputs['output_ownership'];
    if (ownershipTensor != null) {
      final ownershipData = ownershipTensor.data as List<double>? ?? [];
      if (ownershipData.length >= n * n) {
        ownership = List.generate(
          n,
          (r) => List.generate(
            n,
            (c) => _tanh(ownershipData[r * n + c]),
          ),
        );
      }
    }

    return AnalysisResult(
      topMoves: moves,
      winrate: winrate,
      ownership: ownership,
      boardSize: n,
      nextPlayer: board.nextPlayer,
    );
  }

  /// Softmax 函數
  List<double> _softmax(List<double> values) {
    if (values.isEmpty) return [];
    final maxVal = values.reduce(max);
    final exps = values.map((v) => exp(v - maxVal)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sumExps).toList();
  }

  /// Tanh 函數
  double _tanh(double x) {
    final e2x = exp(2 * x);
    return (e2x - 1) / (e2x + 1);
  }

  /// 釋放資源
  Future<void> dispose() async {
    await _session?.close();
    _session = null;
    _isInitialized = false;
  }
}
