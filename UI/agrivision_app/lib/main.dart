import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import 'package:agrivision_app/ml_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const AgriVisionApp());
}

class AgriVisionApp extends StatelessWidget {
  const AgriVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AgriVision',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.green,
        fontFamily: 'Roboto',
      ),
      home: const MainScreen(),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  File? _selectedImage;
  final ImagePicker _picker = ImagePicker();

  final AgriVisionService _mlService = AgriVisionService();
  
  bool _isLoading = false;
  Map<String, dynamic>? _predictionResult;

  // Auto load AI model
  @override
  void initState() {
    super.initState();
    _mlService.loadModel();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
          _isLoading = true;
          _predictionResult = null;
        });

        print("Image loaded: ${pickedFile.path}");

        final result = await _mlService.predict(pickedFile.path);

        setState(() {
          _predictionResult = result;
          _isLoading = false;
        });
      }
    } catch (e) {
      print("ERROR when selecting image: $e");
      setState(() { _isLoading = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: const Color(0xFF5cb85c),
        elevation: 0,
        centerTitle: true,
        title: const Text(
          'AgriVision - Trợ lý Nông nghiệp',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 20,
          ),
        ),
      ),
      // SingleChildScrollView
      body: SingleChildScrollView( 
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image area / Tutorial board
            Container(
              height: 250,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(15),
                border: Border.all(
                  color: const Color(0xFFa5d6a7),
                  width: 2.5,
                ),
              ),
              child: _selectedImage != null
                ? ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.file(
                    _selectedImage!,
                    width: double.infinity,
                    height: double.infinity,
                    fit: BoxFit.cover,
                  ),
                )
                : const Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.image_search_rounded,
                      size: 80,
                      color: Color(0xFFa5d6a7),
                    ),
                    SizedBox(height: 16),
                    Text(
                      'Bấm nút bên dưới để chọn ảnh cây',
                      style: TextStyle(
                        color: Colors.grey,
                        fontSize: 16,
                      ),
                    ),
                  ],
                ),
            ),

            const SizedBox(height: 24),

            // Wrong prediction section
            if (_isLoading)
              const Center(
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: CircularProgressIndicator(color: Color(0xFF4ca64c)),
                ),
              )
            else if (_predictionResult != null)
              Container(
                margin: const EdgeInsets.only(bottom: 24),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: const Color(0xFFf1f8e9),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: const Color(0xFFc5e1a5)),
                ),
                child: Column(
                  children: [
                    const Text(
                      'KẾT QUẢ CHẨN ĐOÁN',
                      style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.grey),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '${_predictionResult!['disease']}',
                      style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Color(0xFFd32f2f)),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Độ tin cậy: ${_predictionResult!['confidence']}%',
                      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500, color: Color(0xFF388e3c)),
                    ),
                  ],
                ),
              ),

            // Open camera button
            ElevatedButton.icon(
              onPressed: () => _pickImage(ImageSource.camera),
              icon: const Icon(Icons.camera_alt, size: 28),
              label: const Text(
                'Mở Máy Ảnh Để Chụp',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF4ca64c),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),

            const SizedBox(height: 16),

            // Choose image from gallery button
            ElevatedButton.icon(
              onPressed: () => _pickImage(ImageSource.gallery),
              icon: const Icon(Icons.photo_library, size: 28),
              label: const Text(
                'Chọn Ảnh Từ Thư Viện',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF2188ff),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
            ),

            const SizedBox(height: 32),

            // History section
            Row(
              children: [
                const Icon(Icons.history, color: Color(0xFF2e7d32), size: 26),
                const SizedBox(width: 8),
                const Text(
                  'Lịch sử tra cứu bệnh cây',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF333333),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 8),
            const Divider(thickness: 1),
            const SizedBox(height: 20),

            // Default empty text
            const Center(
              child: Text(
                'Chưa có lần tra cứu nào.',
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 16,
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}