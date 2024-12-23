package com.example.testtensorflow

import android.content.Context
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainShow : AppCompatActivity() {
    private lateinit var tflite: Interpreter
    private var inp: EditText? = null
    private var outp: TextView? = null
    private var pred: Button? = null
    private var model: String? = "model.tflite"
    private var modelCoice: RadioGroup? = null
    private var m2InputChoice: RadioGroup? = null
    private var m2InputImg: ImageView? = null
    private var m2Scroll: ScrollView? = null
    private var linechart: LineChart? = null
    private val thaiTextToPred = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ0123456789"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main_show)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        init()



        predictButton()

        setupModelChoiceListeners()
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = this.assets.openFd(model!!)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declareLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declareLength)
    }

    private fun setupChart(chart: LineChart, floatArray: FloatArray) {
        // สร้าง List ของจุดข้อมูลจาก FloatArray
        val entries = ArrayList<Entry>()
        for (i in floatArray.indices) {
            entries.add(Entry(i.toFloat(), floatArray[i]))
        }

        // สร้าง LineDataSet
        val dataSet = LineDataSet(entries, "Prediction Results")
        dataSet.color = ContextCompat.getColor(this, android.R.color.holo_blue_dark) // สีของเส้น
        dataSet.lineWidth = 2f // ความหนาของเส้น
        dataSet.setDrawCircles(false) // ไม่วาดวงกลมที่จุดข้อมูล
        dataSet.setDrawValues(false) // ไม่แสดงค่าที่จุดข้อมูล

        // สร้าง LineData และตั้งค่าให้กับกราฟ
        val lineData = LineData(dataSet)
        chart.data = lineData

        // ตั้งค่าฟีเจอร์การซูม
        chart.isDragEnabled = true // เปิดให้ลากกราฟได้
        chart.setScaleEnabled(true) // เปิดให้ซูมได้
        chart.setPinchZoom(true) // เปิดให้ซูมแบบ pinch (ใช้นิ้วสองนิ้ว)

        // ตั้งค่าเพิ่มเติม
        chart.description.isEnabled = false // ปิดคำอธิบายกราฟ
        chart.axisRight.isEnabled = false // ปิดแกน Y ด้านขวา
        chart.xAxis.position = XAxis.XAxisPosition.BOTTOM // แกน X อยู่ด้านล่าง
        chart.xAxis.granularity = 1f // ระยะห่างขั้นต่ำของแกน X
        chart.axisLeft.granularity = 0.1f // ระยะห่างขั้นต่ำของแกน Y

        // รีเฟรชกราฟ
        chart.invalidate()
    }


    private suspend fun loadData(): Array<FloatArray>? {
        return withContext(Dispatchers.IO) {
            try {
                // Read CSV and convert it to Array<FloatArray>
                readCsvAsFloatArray(this@MainShow, "xdata.csv").drop(1).toTypedArray()
            } catch (e: Exception) {
                e.printStackTrace()
                null // Return null if an exception occurs
            }
        }
    }

    private fun predictButton() {
        pred?.setOnClickListener {
            CoroutineScope(Dispatchers.Main).launch {
                try {
                    withContext(Dispatchers.IO) {
                        tflite = Interpreter(loadModelFile())
                    }
                    printModelShape()
                    when (modelCoice?.checkedRadioButtonId) {
                        R.id.option1 -> handleOption1()
                        R.id.option2 -> handleOption2()
                        R.id.option3 -> handleOption3()
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                }
            }
        }
    }

    private fun printModelShape() {
        println("Input shape: ${tflite.getInputTensor(0).shape().joinToString(", ")}")
        println("Output shape: ${tflite.getOutputTensor(0).shape().joinToString(", ")}")
    }

    private fun handleOption1() {
        val inputText = inp?.text?.toString()
        if (inputText.isNullOrBlank()) {
            outp?.text = "Please enter valid input."
            return
        }

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val textSplit = inputText.split(",").map { it.toFloat() }
                val prediction = doInference(textSplit)
                withContext(Dispatchers.Main) {
                    outp?.text = prediction.toString()
                }
            } catch (ex: Exception) {
                ex.printStackTrace()
                withContext(Dispatchers.Main) {
                    outp?.text = "Error during prediction."
                }
            }
        }
    }

    private fun handleOption2() {
        CoroutineScope(Dispatchers.IO).launch {
            try {
                val predictions = predictImage(m2InputImg!!, tflite)
                println(predictions.joinToString { i -> (i*10).toInt().toString() })
                val predClass = predictions.withIndex().maxByOrNull { it.value }?.index
                println(predClass)
                withContext(Dispatchers.Main) {
                    outp?.text = thaiTextToPred[predClass!!.toInt()].toString()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    private fun handleOption3() {
        outp!!.setText("predicting wait a second .....")
        CoroutineScope(Dispatchers.IO).launch {
            val x_pred = loadData() // Assume loadData() returns Array<FloatArray>
            if (x_pred == null || x_pred.isEmpty()) {
                println("x_pred is null or empty")
                return@launch
            }

            val batch_size = 16
            val num_samples = x_pred.size
            val y_pred = FloatArray(num_samples) // Output array

            try {
                println("Input shape: ${x_pred.size} x ${x_pred[0].size}")



                for (i in 0 until (num_samples + batch_size - 1) / batch_size) {
                    val from = i * batch_size
                    val to = minOf((i + 1) * batch_size, num_samples) // Ensure 'to' doesn't exceed the array size

                    val batch = x_pred.copyOfRange(from, to)
                    val y_pred_batch =Array(to - from) { FloatArray(1) }
                    // Run TFLite model inference on the batch
                    tflite.run(batch, y_pred_batch)

                    // Copy the results back to y_pred
                    for (j in 0 until to - from) {
                        y_pred[from + j] = y_pred_batch[j][0] // Copy each row manually
                    }
                }

                // Output predictions for debugging
//                println(y_pred.joinToString { i -> i.toString() })
                setupChart(linechart!!, y_pred)
                outp!!.setText("Predict complete")
//                println("Prediction complete")
            } catch (e: Exception) {
                e.printStackTrace()
                println("Error during prediction: ${e.message}")
            }
        }
    }

    private fun doInference(inputString: List<Float>): Int {
        val inputVal = inputString.toFloatArray()
        val output = Array(1) { FloatArray(1) }
        tflite.run(inputVal, output)
        return output[0][0].toInt()
    }

    private fun preprocessGrayscaleImage(imageView: ImageView): ByteBuffer {
        // Get the bitmap from the ImageView
        val bitmap = (imageView.drawable as BitmapDrawable).bitmap

        // Get the original width and height of the image
        val width = bitmap.width
        val height = bitmap.height

        // Create a ByteBuffer with the shape (1, height, width, 1)
        val byteBuffer = ByteBuffer.allocateDirect(4 * width * height)  // Each float is 4 bytes
        byteBuffer.order(ByteOrder.nativeOrder())  // Set the byte order to native order

        // Create an array to hold the pixel values
        val intValues = IntArray(width * height)
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height)

        // Process each pixel
        var n = 0
        for (pixel in intValues) {
            val r = (pixel shr 16) and 0xFF  // Extract red
            val g = (pixel shr 8) and 0xFF   // Extract green
            val b = pixel and 0xFF           // Extract blue

            // Convert to grayscale (average of R, G, B)
            val grayscale = (r + g + b) / 3.0f / 255.0f  // Normalize to range [0, 1]
//            n++
            if (n++ % 254 == 0) println()
            print("${grayscale}, ")
            // Put the grayscale value into the ByteBuffer
            byteBuffer.putFloat(grayscale)
        }

        // Set position back to 0 to prepare for reading
        byteBuffer.position(0)

        return byteBuffer
    }

    private fun predictImage(image: ImageView, model: Interpreter): FloatArray {
        val inputBuffer = preprocessGrayscaleImage(image)
        val outputArray = Array(1) { FloatArray(55) }
        model.run(inputBuffer, outputArray)

        println(outputArray.size)
        return outputArray[0]
    }

    private fun readCsvAsFloatArray(context: Context, fileName: String): Array<FloatArray> {
        return try {
            context.assets.open(fileName).bufferedReader().useLines { lines ->
                lines.map { line ->
                    line.split(",").map { it.toFloatOrNull() ?: 0.0f }.toFloatArray().copyOfRange(1, 32)
                }.toList().toTypedArray()
            }
        } catch (e: IOException) {
            e.printStackTrace()
            emptyArray()
        }
    }

    private fun setupModelChoiceListeners() {
        modelCoice?.apply {
            check(R.id.option1)
            setOnCheckedChangeListener { _, checkedId ->
                when (checkedId) {
                    R.id.option1 -> setModelUI("model.tflite", true)
                    R.id.option2 -> setModelUI("bbox5.tflite", false)
                    R.id.option3 -> setModelUI("perfect(2).tflite", true)
                }
            }
        }

        m2InputChoice?.setOnCheckedChangeListener { _, checkID ->
            m2InputImg?.setImageResource(
                when (checkID) {
                    R.id.m2InputC1 -> R.drawable.chick1g
                    R.id.m2InputC2 -> R.drawable.chick2
                    R.id.m2InputC3 -> R.drawable.char3
                    R.id.m2InputC4 -> R.drawable.char4
                    else -> R.drawable.default_image
                }
            )
        }
    }

    private fun setModelUI(modelName: String, showInputField: Boolean) {
        model = modelName
        inp?.visibility = if (showInputField) View.VISIBLE else View.INVISIBLE
        m2InputChoice?.visibility = if (showInputField) View.INVISIBLE else View.VISIBLE
        m2InputImg?.visibility = if (showInputField) View.INVISIBLE else View.VISIBLE
        m2Scroll?.visibility = if (showInputField) View.INVISIBLE else View.VISIBLE
    }

    private fun init() {
        inp = findViewById(R.id.m1Input)
        outp = findViewById(R.id.outp)
        pred = findViewById(R.id.pred)
        modelCoice = findViewById(R.id.modelChoice)
        m2InputChoice = findViewById(R.id.m2InputChoice)
        m2InputImg = findViewById(R.id.m2InputImg)
        m2Scroll = findViewById(R.id.m2Scroll)
        linechart = findViewById(R.id.lineChart)
    }
}
