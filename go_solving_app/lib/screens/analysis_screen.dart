import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import '../models/board_state.dart';
import '../models/analysis_result.dart';
import '../services/board_recognition.dart';
import '../services/katago_engine.dart';
import '../widgets/board_painter.dart';
import '../widgets/move_suggestion.dart';

class AnalysisScreen extends StatefulWidget {
  const AnalysisScreen({super.key});

  @override
  State<AnalysisScreen> createState() => _AnalysisScreenState();
}

class _AnalysisScreenState extends State<AnalysisScreen> {
  BoardState? _boardState;
  AnalysisResult? _analysisResult;
  bool _isRecognizing = false;
  bool _isAnalyzing = false;
  String? _errorMessage;
  StoneColor _nextPlayer = StoneColor.black;

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
    setState(() {
      _boardState = recognition.generateSampleBoard();
    });
    _runAnalysis();
  }

  Future<void> _recognizeBoard(String imagePath) async {
    setState(() {
      _isRecognizing = true;
      _errorMessage = null;
    });

    try {
      final recognition = BoardRecognition();
      final boardState = await recognition.recognizeFromImage(imagePath);
      if (recognition.lastDebugInfo != null) {
        debugPrint(recognition.lastDebugInfo.toString());
      }
      if (mounted) {
        setState(() {
          _boardState = boardState;
          _isRecognizing = false;
        });
        _runAnalysis();
      }
    } catch (e, stackTrace) {
      debugPrint('[AnalysisScreen] 辨識錯誤: $e\n$stackTrace');
      if (mounted) {
        setState(() {
          _isRecognizing = false;
          _errorMessage = '棋盤辨識失敗: $e';
        });
      }
    }
  }

  Future<void> _runAnalysis() async {
    if (_boardState == null) return;

    setState(() {
      _isAnalyzing = true;
      _errorMessage = null;
    });

    try {
      final engine = KataGoEngine();
      await engine.initialize();
      final result = await engine.analyze(
        _boardState!.copyWithNextPlayer(_nextPlayer),
      );
      if (mounted) {
        setState(() {
          _analysisResult = result;
          _isAnalyzing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isAnalyzing = false;
          _errorMessage = '分析失敗: $e';
        });
      }
    }
  }

  void _toggleNextPlayer() {
    setState(() {
      _nextPlayer = _nextPlayer == StoneColor.black
          ? StoneColor.white
          : StoneColor.black;
    });
    _runAnalysis();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('分析結果'),
        actions: [
          if (_boardState != null)
            IconButton(
              onPressed: _toggleNextPlayer,
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
            ),
          ),
        ),
        if (_isAnalyzing)
          const Padding(
            padding: EdgeInsets.all(8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                SizedBox(width: 8),
                Text('AI 分析中...'),
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
                child: OutlinedButton.icon(
                  onPressed: _toggleNextPlayer,
                  icon: Container(
                    width: 16,
                    height: 16,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _nextPlayer == StoneColor.black
                          ? Colors.black
                          : Colors.white,
                      border: Border.all(color: Colors.grey),
                    ),
                  ),
                  label: Text(
                    '下一手: ${_nextPlayer == StoneColor.black ? "黑" : "白"}',
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 8),
      ],
    );
  }
}
