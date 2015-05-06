package org.apache.spark.mllib.dlps

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.mllib.linalg.{Vector => SV, Vectors}
import org.apache.spark.mllib.util.{MnistDatasetReader, MnistItem}
import org.apache.spark.rdd.RDD
import org.scalatest.FunSuite

import scala.collection.JavaConversions._

/**
 * Created by maxmin on 3/19/15.
 */

class CassandraTest extends FunSuite with CassandraTestSparkContext {

  test("sc and ps") {

    println("####")
    println(sc.getConf.get("spark.app.name"))

    println("####")
    println("ps keyspace: %s, tablename: %s".format(ps.keyspaceName(), ps.tableName()))

  }

  test("initialization") {
    println("####")
    println("ps initialization")
    ps.initPS(sc.getConf)
  }

  test("random initialize weight matrix") {
    ps.randomInit(1.0, 2.0, 0.5)
  }

  test("merge and update") {
    ps.mergeAndUpdate(1.0)
  }

  def useAccessor(w: Double, b: Double): Double = {
    val psMeta = new PSMeta(Array("10.227.119.245"),
      "test", "modelTest")
    val modelMeta = new ModelMeta(4, Array(10, 5, 5, 3))
    val cred = new PSCredential("psUser", "parameter")
    val accessor = new PSAccessor(modelMeta, psMeta, cred)

    val w1: DenseMatrix[Double] = DenseMatrix.zeros(5, 10)
    val w2: DenseMatrix[Double] = DenseMatrix.zeros(5, 5)
    val w3: DenseMatrix[Double] = DenseMatrix.zeros(3, 5)

    for(i <- 0 until 5)
      for(j <- 0 until 10) {
        w1(i, j) = w
      }
    accessor.setLayerWeightUpdate(1, w1)
    for(i <- 0 until 5)
      for(j <- 0 until 5) {
        w2(i, j) = w
      }
    accessor.setLayerWeightUpdate(2, w2)
    for(i <- 0 until 3)
      for(j <- 0 until 5) {
        w3(i, j) = w
      }
    accessor.setLayerWeightUpdate(3, w3)

    val b1: DenseVector[Double] = DenseVector.zeros(5)
    val b2: DenseVector[Double] = DenseVector.zeros(5)
    val b3: DenseVector[Double] = DenseVector.zeros(3)

    for(i <- 0 until 5)
      b1(i) = b
    accessor.setBiasOfLayer(1, b1)

    for(i <- 0 until 5)
      b2(i) = b
    accessor.setBiasOfLayer(2, b2)

    for(i <- 0 until 3)
      b3(i) = b
    accessor.setBiasOfLayer(3, b3)
    accessor.close()
    w
  }

  def tryTest() {
    val four = Array((1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0))
    val rdd = sc.parallelize(four, 2)
    val mappedRdd = rdd.map(item => {
      val psMeta = new PSMeta(Array("10.227.119.245"),
        "test", "modelTest")
      val modelMeta = new ModelMeta(4, Array(10, 5, 5, 3))
      val cred = new PSCredential("psUser", "parameter")
      val accessor = new PSAccessor(modelMeta, psMeta, cred)
      val w = item._1
      val b = item._2

      val w1: DenseMatrix[Double] = DenseMatrix.zeros(5, 10)
      val w2: DenseMatrix[Double] = DenseMatrix.zeros(5, 5)
      val w3: DenseMatrix[Double] = DenseMatrix.zeros(3, 5)

      for(i <- 0 until 5)
        for(j <- 0 until 10) {
          w1(i, j) = w
        }
      accessor.setLayerWeightUpdate(1, w1)
      for(i <- 0 until 5)
        for(j <- 0 until 5) {
          w2(i, j) = w
        }
      accessor.setLayerWeightUpdate(2, w2)
      for(i <- 0 until 3)
        for(j <- 0 until 5) {
          w3(i, j) = w
        }
      accessor.setLayerWeightUpdate(3, w3)

      val b1: DenseVector[Double] = DenseVector.zeros(5)
      val b2: DenseVector[Double] = DenseVector.zeros(5)
      val b3: DenseVector[Double] = DenseVector.zeros(3)

      for(i <- 0 until 5)
        b1(i) = b
      accessor.setBiasOfLayer(1, b1)

      for(i <- 0 until 5)
        b2(i) = b
      accessor.setBiasOfLayer(2, b2)

      for(i <- 0 until 3)
        b3(i) = b
      accessor.setBiasOfLayer(3, b3)
      accessor.close()
      w
    }
    )
    mappedRdd.collect().foreach( d =>
      println("D: " + d)
    )
  }

  test("one iteration test") {
    tryTest()
  }

  def mnistTrainDataset(size: Int = 10, dropN: Int = 0): (RDD[(SV, SV)], Int) = {
    //val cognginHome = sys.props.getOrElse("cogngin.home", fail("cogngin.home is not set!"))
    val cognginHome = "/Users/maxmin/IdeaProjects/cogngin"
    val labelsFile = s"$cognginHome/data/train-labels-idx1-ubyte.gz"
    val imagesFile = s"$cognginHome/data/train-images-idx3-ubyte.gz"
    val mnistReader = new MnistDatasetReader(labelsFile, imagesFile)
    val numVisible = mnistReader.rows * mnistReader.cols

    // Read the Label Data
    val mnistData = mnistReader.drop(dropN).take(size).map { case m@MnistItem(label, data) =>
      assert(label < 10)
      // Map into the 1*10 Matrix [0,0,0,0,0,0,0,0,0,0]
      val y = DenseVector.zeros[Double](10)
      y(label) = 1
      val x = m.binaryVector
      // Get the Training Data
      (x, Vectors.fromBreeze(y))
    }
    val data: RDD[(SV, SV)] = sc.parallelize(mnistData.toSeq)
    (data, numVisible)
  }

  test("whole system via mllib") {
    val (data, numVisible) = mnistTrainDataset(400)
    data.cache()
    println("input data is cached!")
    //val nn = SimpleMLP.train(data, 20, 10, Array(numVisible, 500, 100, 10), 0.05, 0.0, 0.01)
    val nn = SimpleMLP.train(ps, data, 100, 4, 1, 0.0, 0.01)
    //SimpleMLP.runSGD(data, nn, 37, 6000, 0.1, 0.0, 0.01)
    println("Training Done!")

    val (dataTest, _) = mnistTrainDataset(10000, 100)
    println("Error: " + SimpleMLP.error(dataTest, nn, 100))
    println("====Done Test====/n")
  }
}
