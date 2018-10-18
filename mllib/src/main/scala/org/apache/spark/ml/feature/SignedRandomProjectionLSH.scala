/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.feature

import scala.util.Random

import breeze.linalg.normalize
import org.apache.hadoop.fs.Path

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.HasSeed
import org.apache.spark.ml.util._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType

/**
 * :: Experimental ::
 *
 * Params for [[SignedRandomProjectionLSH]].
 */
private[ml] trait SignedRandomProjectionLSHParams extends Params {

  /**
   * The number of hashes used for AND-amplification.
   *
   * @group param
   */
  val numPlanes: IntParam = new IntParam(this, "numPlanes",
    "the number of hashes used for AND-amplification.",
    ParamValidators.gt(0))

  /** @group getParam */
  final def getNumPlanes: Int = $(numPlanes)
}

/**
 * :: Experimental ::
 *
 * Model produced by [[SignedRandomProjectionLSH]], where multiple random vectors are stored. The
 * vectors are normalized to be unit vectors and each vector is used in a hash function:
 *    `h_i(x) = sign(r_i.dot(x))`
 * where `r_i` is the i-th random unit vector.
 *
 * @param randUnitVectors An array of arrays of random unit vectors representing hash functions.
 */
@Experimental
class SignedRandomProjectionLSHModel private[ml](
    override val uid: String,
    private[ml] val randUnitVectors: Array[Array[Vector]])
  extends LSHModel[SignedRandomProjectionLSHModel] with SignedRandomProjectionLSHParams {

  /** @group setParam */
  override def setInputCol(value: String): this.type = super.set(inputCol, value)

  /** @group setParam */
  override def setOutputCol(value: String): this.type = super.set(outputCol, value)

  override protected[ml] def hashFunction(elems: Vector): Array[Vector] = {
    val hashValues = randUnitVectors.map(
      randUnitVectorsForAndAmplification =>
        Vectors.dense(
          randUnitVectorsForAndAmplification.map(
            randUnitVector => Math.signum(BLAS.dot(elems, randUnitVector))
          )
        )
      )

    hashValues
  }

  override protected[ml] def keyDistance(x: Vector, y: Vector): Double = {
    val denominator = Vectors.norm(x, 2.0) * Vectors.norm(y, 2.0)
    val product = BLAS.dot(x, y)
    val cosineDistance = if (denominator == 0.0) {
      0.0
    } else {
      product/denominator
    }
    Math.acos(cosineDistance) / Math.PI
  }

  override protected[ml] def hashDistance(x: Seq[Vector], y: Seq[Vector]): Double = {
    x.zip(y).map(vectorPair =>
      vectorPair._1.toArray.zip(vectorPair._2.toArray).count(pair => pair._1 != pair._2)
    ).min.toDouble / $(numPlanes)
  }

  override def copy(extra: ParamMap): SignedRandomProjectionLSHModel = {
    val copied = new SignedRandomProjectionLSHModel(uid, randUnitVectors).setParent(parent)
    copyValues(copied, extra)
  }

  override def write: MLWriter = {
    new SignedRandomProjectionLSHModel.SignedRandomProjectionLSHModelWriter(this)
  }
}

/**
 * :: Experimental ::
 *
 * This [[SignedRandomProjectionLSH]] implements Locality Sensitive Hashing functions for
 * Angular distance metrics.
 *
 * The input is dense or sparse vectors, each of which represents a point in Euclidean
 * space. The output will be vectors of configurable dimension. Hash values in the
 * same dimension are calculated by the same hash function.
 *
 * References:
 *
 * 1. <a href="https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Stable_distributions">
 * Wikipedia on Stable Distributions</a>
 *
 * 2. Wang, Jingdong et al. "Hashing for similarity search: A survey." arXiv preprint
 * arXiv:1408.2927 (2014).
 */
@Experimental
class SignedRandomProjectionLSH(override val uid: String)
  extends LSH[SignedRandomProjectionLSHModel]
    with SignedRandomProjectionLSHParams with HasSeed {

  override def setInputCol(value: String): this.type = super.setInputCol(value)

  override def setOutputCol(value: String): this.type = super.setOutputCol(value)

  override def setNumHashTables(value: Int): this.type = super.setNumHashTables(value)

  def this() = {
    this(Identifiable.randomUID("srp-lsh"))
  }

  /** @group setParam */
  def setNumPlanes(value: Int): this.type = set(numPlanes, value)

  /** @group setParam */
  def setSeed(value: Long): this.type = set(seed, value)

  override protected[this] def createRawLSHModel(
    inputDim: Int): SignedRandomProjectionLSHModel = {
    val rand = new Random($(seed))
    val randUnitVectors: Array[Array[Vector]] = {
      Array.fill($(numHashTables)) {
        Array.fill($(numPlanes)) {
          val randArray = Array.fill(inputDim)(rand.nextGaussian())
          Vectors.fromBreeze(normalize(breeze.linalg.Vector(randArray)))
        }
      }
    }
    new SignedRandomProjectionLSHModel(uid, randUnitVectors)
  }

  override def transformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
}

object SignedRandomProjectionLSH extends DefaultParamsReadable[SignedRandomProjectionLSH] {

  override def load(path: String): SignedRandomProjectionLSH = super.load(path)
}

object SignedRandomProjectionLSHModel extends MLReadable[SignedRandomProjectionLSHModel] {

  override def read: MLReader[SignedRandomProjectionLSHModel] = {
    new SignedRandomProjectionLSHModelReader
  }

  override def load(path: String): SignedRandomProjectionLSHModel = super.load(path)

  private[SignedRandomProjectionLSHModel] class SignedRandomProjectionLSHModelWriter(
    instance: SignedRandomProjectionLSHModel) extends MLWriter {

    private case class Data(randUnitVectors: Array[Matrix])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val numRows = instance.randUnitVectors.length
      require(numRows > 0)
      val numCols = instance.randUnitVectors.head.size
      val randMatrices = instance.randUnitVectors.map(
        rUV => Matrices.dense(numRows, numCols, rUV.map(_.toArray).reduce(Array.concat(_, _)))
      )
      val data = Data(randMatrices)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class SignedRandomProjectionLSHModelReader
    extends MLReader[SignedRandomProjectionLSHModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[SignedRandomProjectionLSHModel].getName

    override def load(path: String): SignedRandomProjectionLSHModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val Row(randUnitVectorsMatrix: Array[Matrix]) =
        MLUtils.convertMatrixColumnsToML(data, "randUnitVectors")
        .select("randUnitVectors")
        .head()
      val randUnitVectors = randUnitVectorsMatrix.map(_.rowIter.toArray)
      val model = new SignedRandomProjectionLSHModel(metadata.uid, randUnitVectors)

      metadata.getAndSetParams(model)
      model
    }
  }
}
