package com.example.tiled_app

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import com.github.chrisbanes.photoview.PhotoView

class MainActivity : AppCompatActivity() {
    val REQUEST_CODE_PICK_IMAGE = 100

    // Launch the gallery picker
    private val modelPath = "tiled_11s.tflite"
    private val labelPath = "labels.txt"
    private var interpreter: Interpreter? = null
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build() // preprocess input

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val model = FileUtil.loadMappedFile(this, modelPath)
        val options = Interpreter.Options()
        options.numThreads = 4
        interpreter = Interpreter(model, options)

        val inputShape = interpreter!!.getInputTensor(0).shape()
        val outputShape = interpreter!!.getOutputTensor(0).shape()

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1]
        numElements = outputShape[2]

        val uploadButton = findViewById<Button>(R.id.selectImage)
        uploadButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, REQUEST_CODE_PICK_IMAGE)
        }

    }

    fun tileAndPredict(
        bitmap: Bitmap,
        tileWidth: Int,
        tileHeight: Int,
        interpreter: Interpreter,
        imageProcessor: ImageProcessor
    ): List<BoundingBox> {
        val imageWidth = bitmap.width
        val imageHeight = bitmap.height
        val allDetections = mutableListOf<BoundingBox>()

        // Iterate through tiles
        for (y in 0 until imageHeight step tileHeight) {
            for (x in 0 until imageWidth step tileWidth) {
                // Define the tile boundaries
                val xEnd = (x + tileWidth).coerceAtMost(imageWidth)
                val yEnd = (y + tileHeight).coerceAtMost(imageHeight)

                // Crop the tile
                val tile = Bitmap.createBitmap(bitmap, x, y, xEnd - x, yEnd - y)

                // Preprocess the tile
                val resizedTile = Bitmap.createScaledBitmap(tile, tensorWidth, tensorHeight, false)
                val tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(resizedTile)
                val processedTile = imageProcessor.process(tensorImage)
                val imageBuffer = processedTile.buffer

                // Create output TensorBuffer and run the interpreter
                val output = TensorBuffer.createFixedSize(
                    intArrayOf(1, numChannel, numElements),
                    DataType.FLOAT32
                )
                interpreter.run(imageBuffer, output.buffer)

                // Get detections for the tile
                val detections = bestBox(output.floatArray)

                // Adjust the coordinates of detections for the original image
                detections?.forEach { box ->
                    allDetections.add(
                        box.copy(
                            x1 = (box.x1 * (xEnd - x) + x) / imageWidth,
                            y1 = (box.y1 * (yEnd - y) + y) / imageHeight,
                            x2 = (box.x2 * (xEnd - x) + x) / imageWidth,
                            y2 = (box.y2 * (yEnd - y) + y) / imageHeight
                        )
                    )
                }
            }
        }

        return allDetections
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_CODE_PICK_IMAGE && resultCode == Activity.RESULT_OK) {
            val selectedImageUri: Uri? = data?.data
            selectedImageUri?.let {
                val inputStream: InputStream? = contentResolver.openInputStream(it)
                inputStream?.let { stream ->
                    val bitmap: Bitmap = BitmapFactory.decodeStream(stream)

                    // Tile the image and run predictions
                    val tileWidth = 960
                    val tileHeight = 540
                    val detections = tileAndPredict(bitmap, tileWidth, tileHeight, interpreter!!, imageProcessor)

                    // Draw bounding boxes on the original image
                    val updatedBitmap = drawBoundingBoxes(bitmap, detections)

                    // Display the updated image in the ImageView
                    val imageView = findViewById<PhotoView>(R.id.imageView)
                    imageView.setImageBitmap(updatedBitmap)
                }
            }
        }
    }


    data class BoundingBox(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float,
        val cnf: Float,
        val cls: Int,
        val clsName: String = ""
    )


    private fun bestBox(array: FloatArray) : List<BoundingBox>? {

        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = -1.0f
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                val clsName = "1"
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }


    fun drawBoundingBoxes(bitmap: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }


        for (box in boxes) {
            val rect = RectF(
                box.x1 * mutableBitmap.width,
                box.y1 * mutableBitmap.height,
                box.x2 * mutableBitmap.width,
                box.y2 * mutableBitmap.height
            )
            canvas.drawRect(rect, paint)
        }

        return mutableBitmap
    }
}