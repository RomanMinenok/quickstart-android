// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.firebase.samples.apps.mlkit.java.facedetection;

import android.support.annotation.NonNull;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.samples.apps.mlkit.common.FrameMetadata;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;
import com.google.firebase.samples.apps.mlkit.java.VisionProcessorBase;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/** Face Detector Demo. */
public class FaceDetectionProcessor extends VisionProcessorBase<List<FirebaseVisionFace>> {

  private static final String TAG = "FaceDetectionProcessor";

  private final FirebaseVisionFaceDetector detector;

  private Map<Integer, FaceData> currentFaces = new HashMap<>();

  public FaceDetectionProcessor() {
    FirebaseVisionFaceDetectorOptions options =
        new FirebaseVisionFaceDetectorOptions.Builder()
            .setClassificationMode(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
            .setLandmarkMode(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
            .enableTracking()
            .build();

    detector = FirebaseVision.getInstance().getVisionFaceDetector(options);
  }

  @Override
  public void stop() {
    try {
      detector.close();
    } catch (IOException e) {
      Log.e(TAG, "Exception thrown while trying to close Face Detector: " + e);
    }
  }

  @Override
  protected Task<List<FirebaseVisionFace>> detectInImage(FirebaseVisionImage image) {
    return detector.detectInImage(image);
  }

  @Override
  protected void onSuccess(
      @NonNull List<FirebaseVisionFace> faces,
      @NonNull FrameMetadata frameMetadata,
      @NonNull GraphicOverlay graphicOverlay) {
    removeNoLongerPresentFaces(faces, graphicOverlay);
    processPresentFaces(faces, frameMetadata, graphicOverlay);
  }

  private void processPresentFaces(@NonNull List<FirebaseVisionFace> faces, @NonNull FrameMetadata frameMetadata, @NonNull GraphicOverlay graphicOverlay) {
    for (FirebaseVisionFace face : faces) {
       FaceData existingFaceEntry = currentFaces.get(face.getTrackingId());
       if (existingFaceEntry == null) {
         FaceGraphic faceGraphic = new FaceGraphic(graphicOverlay);
         graphicOverlay.add(faceGraphic);
         existingFaceEntry = new FaceData(face, faceGraphic);
         currentFaces.put(face.getTrackingId(), existingFaceEntry);
       }
       existingFaceEntry.faceGraphic.updateFace(face, frameMetadata.getCameraFacing());
    }
  }

  private void removeNoLongerPresentFaces(@NonNull List<FirebaseVisionFace> faces, @NonNull GraphicOverlay graphicOverlay) {
    for (Iterator<Map.Entry<Integer, FaceData>> it = currentFaces.entrySet().iterator(); it.hasNext();) {
      Map.Entry<Integer, FaceData> mapEntry = it.next();
      boolean found = false;
      for (FirebaseVisionFace face : faces) {
        if (mapEntry.getValue().face.getTrackingId() == face.getTrackingId()) {
          found = true;
          break;
        }
      }
      if (!found) {
        it.remove();
        graphicOverlay.remove(mapEntry.getValue().faceGraphic);
      }
    }
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    Log.e(TAG, "Face detection failed " + e);
  }

  static class FaceData {
    public FirebaseVisionFace face;
    public FaceGraphic faceGraphic;

    public FaceData(FirebaseVisionFace face, FaceGraphic faceGraphic) {
      this.face = face;
      this.faceGraphic = faceGraphic;
    }
  }
}
