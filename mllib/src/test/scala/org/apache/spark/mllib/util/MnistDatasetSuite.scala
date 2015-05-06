/**
 * Created by hank on 12/15/14.
 */

package org.apache.spark.mllib.util

import org.scalatest.{FunSuite, Suite}

import scala.collection.JavaConversions._
import org.apache.spark.mllib.linalg.{Vector => SV}
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, sum => brzSum}
import org.apache.spark.rdd.RDD

trait MnistDatasetSuite extends MLlibTestSparkContext {
  self: Suite =>
  def mnistTrainDataset(size: Int = 10, dropN: Int = 0): (RDD[(SV, SV)], Int) = {
    //val cognginHome = sys.props.getOrElse("cogngin.home", fail("cogngin.home is not set!"))
    val cognginHome = "/home/maxmin/IdeaProjects/cogngin"
    val labelsFile = s"$cognginHome/data/train-labels-idx1-ubyte.gz"
    val imagesFile = s"$cognginHome/data/train-images-idx3-ubyte.gz"
    val mnistReader = new MnistDatasetReader(labelsFile, imagesFile)
    val numVisible = mnistReader.rows * mnistReader.cols

    // Read the Label Data
    val mnistData = mnistReader.drop(dropN).take(size).map { case m@MnistItem(label, data) =>
      assert(label < 10)
      // Map into the 1*10 Matrix [0,0,0,0,0,0,0,0,0,0]
      val y = BDV.zeros[Double](10)
      y(label) = 1
      val x = m.binaryVector
      // Get the Training Data
      (x, Vectors.fromBreeze(y))
    }
    val data: RDD[(SV, SV)] = sc.parallelize(mnistData.toSeq)
    (data, numVisible)
  }
}

