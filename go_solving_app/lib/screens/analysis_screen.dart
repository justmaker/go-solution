import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import '../models/board_state.dart';
import '../models/analysis_result.dart';
import '../services/board_recognition.dart';
import '../services/katago_engine.dart';
import '../services/sgf_service.dart';
import '../widgets/board_painter.dart';
import '../widgets/move_suggestion.dart';

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  BoardState? _boardState;
  BoardState? _initialBoardState;
  final List<BoardState> _history = [];
  final KataGoEngine _engine = KataGoEngine();
  final SgfService _sgfService = SgfService();
  StreamSubscription<AnalysisResult>? _analysisSubscription;

  AnalysisResult? _analysisResult;
  bool _isRecognizing = false;
  bool _isAnalyzing = false;
  bool _isExporting = false;
  String? _errorMessage;
  StoneColor _nextPlayer = StoneColor.black;

  @override
  void dispose() {
    _analysisSubscription?.cancel();
    _engine.dispose();
    super.dispose();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final imagePath = ModalRoute.of(context)?.settings.arguments as String?;
    if (imagePath != null && _boardState == null && !_isRecognizing) {
      if (imagePath == 'demo') {
        _loadDemoBoard();
      } else {
        _recognizeBoard(imagePath);
      }
    }
  }

  void _loadDemoBoard() {
    final recognition = BoardRecognition();
    final board = recognition.generateSampleBoard();
    setState(() {
      _boardState = board;
      _initialBoardState = board;
      _history.clear();
      _nextPlayer = board.nextPlayer;
      _analysisResult = null;
    });
    // 選取/載入棋盤後不自動分析，等使用者按下「分析」按鈕
  }

  Future<void> _recognizeBoard(String imagePath) async {
    setState(() {
      _isRecognizing = true;
      _errorMessage = null;
    });

    try {
      final recognition = BoardRecognition();
      final boardState = await recognition.recognizeFromImage(imagePath);
      if (kDebugMode && recognition.lastDebugInfo != null) {
        debugPrint(recognition.lastDebugInfo.toString());
      }
      if (mounted) {
        setState(() {
          _boardState = boardState;
          _initialBoardState = boardState;
          _history.clear();
          _nextPlayer = boardState.nextPlayer;
          _isRecognizing = false;
          _analysisResult = null;
        });
        // 棋盤辨識完成後不自動分析，等使用者按下「分析」按鈕才開始
      }
    } catch (e, stackTrace) {
      if (kDebugMode) {
        debugPrint('[AnalysisScreen] 辨識錯誤: $e\n$stackTrace');
      }
      if (mounted) {
        setState(() {
          _isRecognizing = false;
          _errorMessage = '棋盤辨識失敗: $e';
        });
      }
    }
  }

  Future<void> _runAnalysis() async {
    if (_boardState == null || _isAnalyzing) return;

    // 取消目前的分析（理論上不該發生，因為上面已擋掉重複呼叫）
    await _analysisSubscription?.cancel();
    _engine.cancelAnalysis();

    setState(() {
      _isAnalyzing = true;
      _errorMessage = null;
      _analysisResult = null;
    });

    try {
      if (!_engine.isInitialized) {
        await _engine.initialize();
      }

      final stateToAnalyze = _boardState!.copyWithNextPlayer(_nextPlayer);

      _analysisSubscription = _engine.analyze(stateToAnalyze).listen(
        (result) {
          if (mounted) {
            setState(() {
              _analysisResult = result;
            });
          }
        },
        onError: (e) {
          if (mounted) {
            setState(() {
              _isAnalyzing = false;
              _errorMessage = '分析失敗: $e';
            });
          }
        },
        onDone: () {
          if (mounted) {
            setState(() {
              _isAnalyzing = false;
            });
          }
        },
      );
    } catch (e) {
      if (mounted) {
        setState(() {
          _isAnalyzing = false;
          _errorMessage = '分析失敗: $e';
        });
      }
    }
  }

  Future<void> _stopAnalysis() async {
    _engine.cancelAnalysis();
    await _analysisSubscription?.cancel();
    _analysisSubscription = null;
    if (mounted) {
      setState(() {
        _isAnalyzing = false;
      });
    }
  }

  void _toggleNextPlayer() {
    if (_isAnalyzing) return;
    setState(() {
      _nextPlayer = _nextPlayer == StoneColor.black
          ? StoneColor.white
          : StoneColor.black;
      _analysisResult = null;
    });
  }

  void _onBoardTap(int row, int col) {
    if (_boardState == null || _isAnalyzing) return;

    try {
      // Check if empty
      if (_boardState!.getStone(row, col) != StoneColor.empty) return;

      setState(() {
        _history.add(_boardState!);
        _boardState = _boardState!.playMove(row, col);
        _nextPlayer = _boardState!.nextPlayer;
        _analysisResult = null;
      });
    } catch (e) {
      if (kDebugMode) {
        debugPrint('Invalid move: $e');
      }
    }
  }

  void _undo() {
    if (_history.isEmpty || _isAnalyzing) return;
    setState(() {
      _boardState = _history.removeLast();
      _nextPlayer = _boardState!.nextPlayer;
      _analysisResult = null;
    });
  }

  void _clear() {
    if (_initialBoardState == null || _isAnalyzing) return;
    setState(() {
      _boardState = _initialBoardState;
      _history.clear();
      _nextPlayer = _boardState!.nextPlayer;
      _analysisResult = null;
    });
  }

  Future<void> _exportSgf() async {
    if (_boardState == null || _isExporting) return;

    setState(() {
      _isExporting = true;
    });

    try {
      // 匯出「目前畫面上看到的盤面」，不論實際下子順序為何，
      // SGF 內容一律以 setup properties (AB/AW) 記錄最終盤面的所有棋子位置，
      // 確保匯出結果與匯出當下畫面一致。
      final boardToExport = _boardState!.copyWithNextPlayer(_nextPlayer);
      final file = await _sgfService.exportToFile(boardToExport);
      if (mounted) {
        await Share.shareXFiles(
          [XFile(file.path)],
          text: '圍棋盤面 SGF 匯出',
        );
      }
    } catch (e) {
      if (kDebugMode) {
        debugPrint('[AnalysisScreen] SGF 匯出失敗: $e');
      }
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('SGF 匯出失敗：$e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isExporting = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('分析結果'),
        actions: [
          if (_boardState != null)
            IconButton(
              onPressed: _isExporting ? null : _exportSgf,
              icon: _isExporting
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.file_download),
              tooltip: '匯出 SGF',
            ),
          if (_boardState != null)
            IconButton(
              onPressed: _isAnalyzing ? null : _toggleNextPlayer,
              icon: Icon(
                Icons.swap_horiz,
                color: _nextPlayer == StoneColor.black
                    ? Colors.black
                    : Colors.white,
              ),
              tooltip: '切換下一手: ${_nextPlayer == StoneColor.black ? "黑" : "白"}',
            ),
        ],
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_isRecognizing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('正在辨識棋盤...'),
          ],
        ),
      );
    }

    if (_errorMessage != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 64, color: Colors.red),
              const SizedBox(height: 16),
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 24),
              FilledButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('返回'),
              ),
            ],
          ),
        ),
      );
    }

    if (_boardState == null) {
      return const Center(child: Text('無棋盤資料'));
    }

    return Column(
      children: [
        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: BoardPainterWidget(
              boardState: _boardState!,
              analysisResult: _analysisResult,
              onBoardTap: _onBoardTap,
            ),
          ),
        ),
        if (_isAnalyzing)
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                const SizedBox(width: 8),
                Text(_analysisResult != null && _analysisResult!.maxVisits > 0
                    ? 'AI 分析中... (${_analysisResult!.visits}/${_analysisResult!.maxVisits})'
                    : 'AI 分析中...'),
              ],
            ),
          ),
        if (_analysisResult != null) ...[
          MoveSuggestionWidget(result: _analysisResult!),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Chip(
                  avatar: const CircleAvatar(
                    backgroundColor: Colors.black,
                    radius: 8,
                  ),
                  label: Text(
                    '黑 ${(_analysisResult!.blackWinrate * 100).toStringAsFixed(1)}%',
                  ),
                ),
                const SizedBox(width: 16),
                Chip(
                  avatar: const CircleAvatar(
                    backgroundColor: Colors.white,
                    radius: 8,
                  ),
                  label: Text(
                    '白 ${(_analysisResult!.whiteWinrate * 100).toStringAsFixed(1)}%',
                  ),
                ),
              ],
            ),
          ),
        ],
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            children: [
              Expanded(
                child: _isAnalyzing
                    ? FilledButton.icon(
                        onPressed: _stopAnalysis,
                        style: FilledButton.styleFrom(
                          backgroundColor: Colors.red,
                        ),
                        icon: const Icon(Icons.stop),
                        label: const Text('停止分析'),
                      )
                    : FilledButton.icon(
                        onPressed: _runAnalysis,
                        icon: const Icon(Icons.auto_awesome),
                        label: const Text('分析'),
                      ),
              ),
              const SizedBox(width: 8),
              OutlinedButton.icon(
                onPressed: (_history.isEmpty || _isAnalyzing) ? null : _undo,
                icon: const Icon(Icons.undo),
                label: const Text('復原'),
              ),
              const SizedBox(width: 8),
              OutlinedButton.icon(
                onPressed: ((_history.isEmpty &&
                            _boardState == _initialBoardState) ||
                        _isAnalyzing)
                    ? null
                    : _clear,
                icon: const Icon(Icons.refresh),
                label: const Text('重置'),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
      ],
    );
  }
}
