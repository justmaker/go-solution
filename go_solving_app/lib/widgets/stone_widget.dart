import 'package:flutter/material.dart';
import '../models/board_state.dart';

/// 單個棋子 Widget
class StoneWidget extends StatelessWidget {
  final StoneColor color;
  final double size;
  final String? label;

  const StoneWidget({
    super.key,
    required this.color,
    this.size = 24,
    this.label,
  });

  @override
  Widget build(BuildContext context) {
    if (color == StoneColor.empty) {
      return SizedBox(width: size, height: size);
    }

    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color == StoneColor.black ? Colors.black : Colors.white,
        border: Border.all(
          color: color == StoneColor.black ? Colors.black : Colors.grey,
          width: 1,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            blurRadius: 2,
            offset: const Offset(1, 1),
          ),
        ],
      ),
      child: label != null
          ? Center(
              child: Text(
                label!,
                style: TextStyle(
                  color: color == StoneColor.black ? Colors.white : Colors.black,
                  fontSize: size * 0.5,
                  fontWeight: FontWeight.bold,
                ),
              ),
            )
          : null,
    );
  }
}
