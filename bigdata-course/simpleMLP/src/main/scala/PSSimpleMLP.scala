import org.apache.spark.mllib.dl.{SimpleMLP => MLP}
import org.apache.spark.mllib.dlps.{CassandraPS, SimpleMLP => PSMLP}
import org.apache.spark.mllib.linalg.{Vector => SV, Vectors}
import org.apache.spark.mllib.util.{MnistDatasetReader, MnistItem}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConversions._

/**
 * Created by maxmin on 4/17/15.
 */

object PSSimpleMLP {

  def mnistTrainDataset(sc: SparkContext, imagesFile: String, labelsFile: String,
                        size: Int = 10, dropN: Int = 0): (RDD[(SV, SV)], Int) = {
    //val cognginHome = sys.props.getOrElse("cogngin.home", fail("cogngin.home is not set!"))
    val mnistReader = new MnistDatasetReader(labelsFile, imagesFile)
    val numVisible = mnistReader.rows * mnistReader.cols

    // Read the Label Data
    val rawData = mnistReader.drop(dropN).take(size)
    val rawRDD = sc.parallelize(rawData.toSeq)
    val data = rawRDD.map { case m@MnistItem(label, data) =>
      assert(label < 10)
      // Map into the 1*10 Matrix [0,0,0,0,0,0,0,0,0,0]
      val y = Array.fill[Double](10){0.0}
      y(label) = 1.0
      val x = m.binaryVector
      // Get the Training Data
      (x, Vectors.dense(y))
    }
    //val data: RDD[(SV, SV)] = sc.parallelize(mnistData.toSeq)
    (data, numVisible)
  }

  def main(args: Array[String]): Unit = {

    if (args.length < 6) {
      System.err.println("Usage: SimpleMLPTest <mode> <train> <label> " +
        "<totalSample> <batchSize> " +
        "<totalIterations>")
      System.exit(1)
    }

    val mode = args(0).toInt == 1


    val images =
      if (args(1).equals("default")) {
        "/home/maxmin/bigdata-course/cogngin/data/train-images-idx3-ubyte.gz"
      }
      else {
        args(1)
      }
    val labels =
      if (args(2).equals("default")) {
        "/home/maxmin/bigdata-course/cogngin/data/train-labels-idx1-ubyte.gz"
      }
      else {
        args(2)
      }
    val conf = new SparkConf().setAppName("SimpleMLPTest")
    val start = System.currentTimeMillis()
    if(mode) {
      val sc = new SparkContext(conf)
      println("#### Training samples %s".format(images))
      println("#### Training labels %s".format(labels))
      val (data, numVisible) = mnistTrainDataset(sc, images, labels, args(3).toInt)
      data.cache()
      println("#### Total data number: %d".format(data.count()))
      println("#### Input data is cached!")
      val nn = MLP.train(data, args(4).toInt, args(5).toInt, Array(numVisible, 400, 100, 10), 1, 0.0, 0.01)
      //val nn = SimpleMLP.train(data, 20, 10, Array(numVisible, 500, 100, 10), 0.05, 0.0, 0.01)
      val end = System.currentTimeMillis()
      println("#### Training Done! Time: %s in ms".format(end-start))

      val (dataTest, _) = mnistTrainDataset(sc, images, labels, 200, 400)
      println("Error: " + MLP.error(dataTest, nn, 100))
      println("====Done Test====/n")
    }
    else {
      val (sc, ps) = CassandraPS(conf, 4, Array(784, 400, 100, 10))
      println("#### Training samples %s".format(images))
      println("#### Training labels %s".format(labels))
      val (data, numVisible) = mnistTrainDataset(sc, images, labels, args(3).toInt)
      data.cache()
      println("#### Total data number: %d".format(data.count()))
      println("#### Input data is cached!")
      val nn = PSMLP.train(ps, data, args(4).toInt, args(5).toInt, 1, 0.0, 0.01)
      val end = System.currentTimeMillis()
      println("#### Training Done! Time: %s in ms".format(end-start))

      val (dataTest, _) = mnistTrainDataset(sc, images, labels, 50000, 1000)
      println("Error: " + PSMLP.error(dataTest, nn, 100))
      println("====Done Test====/n")
    }
  }
}
