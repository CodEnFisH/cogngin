package org.apache.spark.mllib.dlps

import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, Suite}

/**
 * Created by maxmin on 3/19/15.
 */

trait CassandraTestSparkContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _
  @transient var ps: CassandraPS = _
  // Read the Spark Configuration
  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("CassandraTest")
    val (local_sc,local_ps) = CassandraPS(conf, 4, Array(784, 500, 100, 10))
    sc = local_sc
    ps = local_ps
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    super.afterAll()
  }
}
