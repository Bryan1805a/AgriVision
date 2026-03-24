import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math; // Thêm thư viện toán học
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class AgriVisionService {
  Interpreter? _interpreter;
  List<String>? _labels;

  bool get isModelLoaded => _interpreter != null;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/agrivision_model_float32.tflite');
      
      _labels = [
        'Ngô - Đốm lá xám (Cercospora)', 
        'Ngô - Rỉ sắt (Common rust)',
        'Ngô - Cháy lá (Northern Leaf Blight)',
        'Ngô - Khỏe mạnh',
        'Ớt chuông - Đốm vi khuẩn',
        'Ớt chuông - Khỏe mạnh',
        'Khoai tây - Bệnh mốc sương sớm',
        'Khoai tây - Bệnh mốc sương mai',
        'Khoai tây - Khỏe mạnh',
        'Cà chua - Đốm vi khuẩn',
        'Cà chua - Bệnh mốc sương sớm',
        'Cà chua - Bệnh mốc sương mai',
        'Cà chua - Nấm mốc lá',
        'Cà chua - Đốm lá Septoria',
        'Cà chua - Nhện đỏ',
        'Cà chua - Đốm vòng (Target Spot)',
        'Cà chua - Xoăn lá vàng do Virus',
        'Cà chua - Khảm lá do Virus',
        'Cà chua - Khỏe mạnh'
      ];
      print('>>> Đã nạp thành công bộ não AgriVision!');
    } catch (e) {
      print('>>> Lỗi khi nạp model: $e');
    }
  }

  Future<Map<String, dynamic>?> predict(String imagePath) async {
    if (_interpreter == null) return null;

    File imageFile = File(imagePath);
    img.Image? originalImage = img.decodeImage(imageFile.readAsBytesSync());
    if (originalImage == null) return null;

    // 1. Tự động kiểm tra xem Model cần NCHW (PyTorch gốc) hay NHWC (TensorFlow mặc định)
    var inputShape = _interpreter!.getInputTensor(0).shape;
    bool isNCHW = inputShape[1] == 3; // Nếu chiều thứ 2 là 3 kênh màu -> NCHW

    // 2. Tiền xử lý ảnh theo đúng chuẩn mô hình yêu cầu
    Float32List inputBytes = preprocessImage(originalImage, isNCHW);
    var input = inputBytes.reshape(inputShape); // Ép khuôn linh hoạt
    
    var outputShape = _interpreter!.getOutputTensor(0).shape; // Thường là [1, 19]
    var output = List.filled(outputShape.reduce((a, b) => a * b), 0.0).reshape(outputShape);

    // Chạy suy luận
    _interpreter!.run(input, output);

    // Lấy mảng điểm số thô (Logits)
    List<double> logits = (output[0] as List).cast<double>();

    // 3. Áp dụng công thức Softmax để tính % xác suất chính xác
    double maxLogit = logits.reduce(math.max); // Tránh tràn số (overflow)
    double sumExp = 0.0;
    List<double> probabilities = List.filled(logits.length, 0.0);
    
    for (int i = 0; i < logits.length; i++) {
      probabilities[i] = math.exp(logits[i] - maxLogit);
      sumExp += probabilities[i];
    }

    double maxProb = 0.0;
    int maxIndex = -1;
    
    for (int i = 0; i < probabilities.length; i++) {
      probabilities[i] /= sumExp; // Hoàn thiện Softmax
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    return {
      'disease': _labels![maxIndex],
      'confidence': (maxProb * 100).toStringAsFixed(2),
    };
  }

  /// Hàm chuẩn hóa pixel, hỗ trợ cả 2 định dạng bộ nhớ
  /// Hàm tiền xử lý ảnh (Resize, Center Crop & Normalize ImageNet)
  Float32List preprocessImage(img.Image originalImage, bool isNCHW) {
    // 1. CHỐNG BÓP MÉO: Cắt ảnh thành hình vuông ở chính giữa (Center Crop)
    int size = math.min(originalImage.width, originalImage.height);
    int offsetX = (originalImage.width - size) ~/ 2;
    int offsetY = (originalImage.height - size) ~/ 2;
    img.Image squareImage = img.copyCrop(originalImage, x: offsetX, y: offsetY, width: size, height: size);
    
    // Resize 224x224 EfficientNet
    img.Image resizedImage = img.copyResize(squareImage, width: 224, height: 224);

    var float32Data = Float32List(1 * 3 * 224 * 224);
    final mean = [0.485, 0.456, 0.406];
    final std = [0.229, 0.224, 0.225];

    int pixelIndex = 0;
    int rOffset = 0;
    int gOffset = 224 * 224;
    int bOffset = 2 * 224 * 224;

    for (int y = 0; y < 224; y++) {
      for (int x = 0; x < 224; x++) {
        var pixel = resizedImage.getPixel(x, y);

        // Normalized
        double r = pixel.rNormalized.toDouble();
        double g = pixel.gNormalized.toDouble();
        double b = pixel.bNormalized.toDouble();

        // NẾU MODEL CHẠY SAI BÉT THÌ MỞ 3 COMMENT NÀY:
        double temp = r;
        r = b;
        b = temp;

        // Standardlised ImageNet
        double rNorm = (r - mean[0]) / std[0];
        double gNorm = (g - mean[1]) / std[1];
        double bNorm = (b - mean[2]) / std[2];

        // List sorting according TFLite
        if (isNCHW) {
          float32Data[rOffset++] = rNorm;
          float32Data[gOffset++] = gNorm;
          float32Data[bOffset++] = bNorm;
        } else {
          float32Data[pixelIndex++] = rNorm;
          float32Data[pixelIndex++] = gNorm;
          float32Data[pixelIndex++] = bNorm;
        }
      }
    }
    return float32Data;
  }
}