from fastapi import HTTPException
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import joblib

from utils.schema import ClassificationFeatures

class ClassificationService:
    def __init__(self):
        load_dotenv()
        model_single_path = os.getenv("MODEL_SINGLE_PATH", "unknown.pt") # biarkan untuk jadi opsi
        model_stack_path = os.getenv("MODEL_STACK_PATH")
        scaler_path = os.getenv("SCALER_PATH")
        encoder_path = os.getenv("ENCODER_LABEL_PATH")

        self.model = joblib.load(model_stack_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)

        self.exclude_cols = [
            "submission_rating",
            "tracking_status",
            "submission_status",
        ]

    async def analysis(self, params: ClassificationFeatures):
        try:
            params_dict = params.model_dump()

            # ==========================
            #   1. Konversi datetime → UNIX
            # ==========================
            datetime_cols = [
                "tracking_first_opened_at",
                "tracking_completed_at",
                "completion_created_at",
                "submission_created_at",
                "submission_ended_review_at",
            ]

            for col in datetime_cols:
                ts = pd.to_datetime(params_dict[col], errors="coerce")
                if pd.isna(ts):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Format datetime salah pada field '{col}'"
                    )
                params_dict[col] = int(ts.value // 10**9)

            # ==========================
            #   2. Buat DataFrame
            # ==========================
            X = pd.DataFrame([params_dict])

            # cek apakah kolom sudah lengkap
            expected_scaled_cols = list(self.scaler.feature_names_in_)
            all_expected = expected_scaled_cols + self.exclude_cols

            missing_cols = [c for c in all_expected if c not in X.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input kurang kolom: {missing_cols}"
                )

            # ==========================
            #   3. Scaling (persis seperti training)
            # ==========================
            X_to_scale = X[expected_scaled_cols]          
            X_not_scaled = X[self.exclude_cols]          

            try:
                X_scaled = self.scaler.transform(X_to_scale)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gagal scaling: {str(e)}"
                )

            # gabungkan kembali
            X_scaled_df = pd.DataFrame(X_scaled, columns=expected_scaled_cols)
            X_final = pd.concat([X_scaled_df, X_not_scaled], axis=1)

            # OUTPUT columns harus sesuai training
            X_final = X_final[self.model.feature_names_in_]


            # ==========================
            #   4. Prediksi
            # ==========================
            pred = self.model.predict(X_final)

            try:
                label = self.encoder.inverse_transform(pred)[0]
            except Exception:
                label = str(pred[0])  # fallback aman

            return {
                "predicted_label": label,
                "raw_output": float(pred[0]),
                "processed_features": params_dict,
            }

        except HTTPException:
            raise

        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Service gagal: {str(e)}"
            )
