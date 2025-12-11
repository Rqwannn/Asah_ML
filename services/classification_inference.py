from fastapi import HTTPException
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import joblib
from utils.schema import ClassificationFeatures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import lime
from lime import lime_tabular
from PIL import Image

class ClassificationService:
    def __init__(self):
        load_dotenv()
        model_single_path = os.getenv("MODEL_SINGLE_PATH", "unknown.pt")
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
        
        self.feature_descriptions = {
            "tracking_status": "Status pelacakan pembelajaran",
            "tracking_first_opened_at": "Waktu pertama kali membuka materi",
            "tracking_completed_at": "Waktu menyelesaikan pembelajaran",
            "completion_created_at": "Waktu pencatatan penyelesaian",
            "completion_enrolling_times": "Jumlah kali mendaftar",
            "completion_study_duration": "Durasi belajar (menit)",
            "completion_avg_submission_rating": "Rata-rata rating tugas",
            "submission_status": "Status pengumpulan tugas",
            "submission_created_at": "Waktu pengumpulan tugas",
            "submission_duration": "Durasi pengerjaan tugas (menit)",
            "submission_ended_review_at": "Waktu selesai review tugas",
            "submission_rating": "Rating tugas"
        }

    async def analysis(self, params: ClassificationFeatures):
        try:
            params_dict = params.model_dump()
            original_params = params_dict.copy()
            
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
            
            X = pd.DataFrame([params_dict])
            
            expected_scaled_cols = list(self.scaler.feature_names_in_)
            all_expected = expected_scaled_cols + self.exclude_cols
            missing_cols = [c for c in all_expected if c not in X.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input kurang kolom: {missing_cols}"
                )
            
            X_to_scale = X[expected_scaled_cols]          
            X_not_scaled = X[self.exclude_cols]
            
            X_original = pd.concat([X_to_scale, X_not_scaled], axis=1)
            X_original = X_original[expected_scaled_cols + self.exclude_cols]
            
            try:
                X_scaled = self.scaler.transform(X_to_scale)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Gagal scaling: {str(e)}"
                )
            
            X_scaled_df = pd.DataFrame(X_scaled, columns=expected_scaled_cols)
            X_final = pd.concat([X_scaled_df, X_not_scaled], axis=1)
            
            X_final = X_final[self.model.feature_names_in_]
            
            pred = self.model.predict(X_final)
            pred_proba = self.model.predict_proba(X_final)
            
            try:
                label = self.encoder.inverse_transform(pred)[0]
            except Exception:
                label = str(pred[0])
            
            def predict_fn_wrapper(data):
                df = pd.DataFrame(data, columns=X_final.columns)
                return self.model.predict_proba(df)
            
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_final.values,
                feature_names=X_final.columns.tolist(),
                class_names=self.encoder.classes_.tolist(),
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )
            
            pred_class_idx = np.argmax(pred_proba[0])
            
            explanation = explainer.explain_instance(
                data_row=X_final.values[0],
                predict_fn=predict_fn_wrapper,
                num_features=len(X_final.columns),
                labels=(pred_class_idx,)
            )
            
            feature_importance = {}
            for feature in X_final.columns:
                original_val = X_final[feature].values[0]
                
                X_test = X_final.copy()
                X_test[feature] = X_test[feature].mean()
                
                prob_original = pred_proba[0][pred_class_idx]
                prob_changed = self.model.predict_proba(X_test)[0][pred_class_idx]
                
                importance = prob_original - prob_changed
                feature_importance[feature] = importance
            
            viz1_base64 = self._create_lime_feature_importance_viz(
                explanation, label, pred_proba, self.encoder.classes_, 
                feature_importance, pred_class_idx
            )
            viz2_base64 = self._create_student_friendly_dashboard(
                explanation, label, pred_proba, self.encoder.classes_, 
                original_params, feature_importance, pred_class_idx, X_original
            )

            # =================================================
            # DEBUGG Menampilkan visualisasinya
            # =================================================

            # viz1_image_data = base64.b64decode(viz1_base64)
            # viz1_image = Image.open(BytesIO(viz1_image_data))
            # viz1_image.show()

            # viz2_image_data = base64.b64decode(viz2_base64)
            # viz2_image = Image.open(BytesIO(viz2_image_data))
            # viz2_image.show()

            # with open("debug_lime_visualization.png", "wb") as f:
            #     f.write(viz1_image_data)

            # with open("debug_confidence_visualization.png", "wb") as f:
            #     f.write(viz2_image_data)

            # print("Visualisasi LIME disimpan ke: debug_lime_visualization.png")
            # print("Visualisasi Dashboard disimpan ke: debug_confidence_visualization.png")
            # print(f"Feature Importance: {feature_importance}")
            
            return {
                "predicted_label": label,
                "raw_output": float(pred[0]),
                "processed_features": params_dict,
                "prediction_probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.encoder.classes_, pred_proba[0])
                },
                "feature_importance": {k: float(v) for k, v in feature_importance.items()},
                "lime_visualization": viz1_base64,
                "confidence_visualization": viz2_base64
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Service gagal: {str(e)}"
            )
    
    def _create_lime_feature_importance_viz(self, explanation, predicted_label, pred_proba, class_names, feature_importance, pred_class_idx):
        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1], hspace=0.35, wspace=0.35, 
                             left=0.08, right=0.96, top=0.92, bottom=0.08)
        
        ax1 = fig.add_subplot(gs[0, :])
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        features = []
        weights = []
        
        for feature_name, weight in sorted_features:
            if feature_name in self.feature_descriptions:
                display_name = self.feature_descriptions[feature_name]
            else:
                display_name = feature_name
            
            features.append(display_name)
            weights.append(weight)
        
        colors = ['#27ae60' if w > 0 else '#e74c3c' for w in weights]
        
        y_pos = np.arange(len(features))
        bars = ax1.barh(y_pos, weights, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5, height=0.7)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(features, fontsize=11, fontweight='bold')
        ax1.set_xlabel('Pengaruh Fitur terhadap Prediksi', fontsize=13, fontweight='bold', labelpad=10)
        ax1.set_title(f'10 Fitur Paling Berpengaruh untuk Prediksi: {predicted_label}', 
                     fontsize=15, fontweight='bold', pad=20)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax1.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
        
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            label_text = f'{weight:.4f}'
            x_pos = weight + (0.003 if weight > 0 else -0.003)
            ax1.text(x_pos, i, label_text, 
                   va='center', ha='left' if weight > 0 else 'right',
                   fontsize=10, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
        
        positive_patch = plt.Rectangle((0, 0), 1, 1, fc='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.5)
        negative_patch = plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.5)
        ax1.legend([positive_patch, negative_patch], 
                  ['Meningkatkan kepercayaan prediksi', 'Menurunkan kepercayaan prediksi'],
                  loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')
        
        ax2 = fig.add_subplot(gs[1, 0])
        
        probabilities = pred_proba[0]
        colors_bar = ['#3498db' if cls != predicted_label else '#e74c3c' 
                      for cls in class_names]
        
        bars = ax2.bar(range(len(class_names)), probabilities, 
                       color=colors_bar, alpha=0.85, edgecolor='black', linewidth=2, width=0.6)
        ax2.set_xticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=0, ha='center', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Tingkat Kepercayaan', fontsize=12, fontweight='bold', labelpad=10)
        ax2.set_title('Tingkat Kepercayaan untuk Setiap Kategori', 
                      fontsize=13, fontweight='bold', pad=15)
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        
        for i, (bar, prob) in enumerate(zip(bars, probabilities)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.04,
                    f'{prob:.1%}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        ax3 = fig.add_subplot(gs[1, 1])
        
        explode = [0.12 if cls == predicted_label else 0 for cls in class_names]
        colors_pie = ['#e74c3c' if cls == predicted_label else '#95a5a6' 
                      for cls in class_names]
        
        wedges, texts, autotexts = ax3.pie(
            probabilities, 
            labels=class_names,
            autopct='%1.1f%%',
            startangle=90,
            explode=explode,
            colors=colors_pie,
            textprops={'fontsize': 12, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 2.5}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(13)
            autotext.set_fontweight('bold')
        
        ax3.set_title(f'Distribusi Prediksi\nHasil: {predicted_label}', 
                      fontsize=13, fontweight='bold', pad=15)
        
        fig.suptitle('Analisis LIME: Pengaruh Setiap Fitur terhadap Prediksi Model', 
                    fontsize=17, fontweight='bold', y=0.97)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
    
    def _create_student_friendly_dashboard(self, explanation, predicted_label, pred_proba, class_names, params, feature_importance, pred_class_idx, X_original):
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[0.6, 1.5, 1], hspace=0.4, wspace=0.3, 
                             left=0.07, right=0.97, top=0.92, bottom=0.06)
        
        ax_header = fig.add_subplot(gs[0, :])
        ax_header.axis('off')
        
        max_prob = np.max(pred_proba[0])
        confidence_text = "Sangat Yakin" if max_prob > 0.9 else "Yakin" if max_prob > 0.7 else "Cukup Yakin" if max_prob > 0.5 else "Kurang Yakin"
        confidence_color = "#27ae60" if max_prob > 0.7 else "#f39c12" if max_prob > 0.5 else "#e74c3c"
        
        header_text = f"""HASIL ANALISIS PEMBELAJARAN
        
Kategori Prediksi: {predicted_label}  |  Tingkat Keyakinan: {confidence_text} ({max_prob:.1%})

Model AI telah menganalisis pola belajar Anda dan memprediksi kategori "{predicted_label}"
        """
        
        ax_header.text(0.5, 0.5, header_text, 
                      ha='center', va='center', fontsize=13, 
                      bbox=dict(boxstyle='round,pad=1.2', facecolor=confidence_color, alpha=0.2, 
                               edgecolor=confidence_color, linewidth=3),
                      fontweight='bold', linespacing=1.8)
        
        ax_factors = fig.add_subplot(gs[1, :])
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        
        factor_names = []
        factor_values = []
        factor_colors = []
        factor_explanations = []
        
        for feature_name, weight in sorted_features:
            if feature_name in self.feature_descriptions:
                desc = self.feature_descriptions[feature_name]
            else:
                desc = feature_name
            
            factor_names.append(desc)
            factor_values.append(abs(weight))
            
            if weight > 0:
                factor_colors.append('#27ae60')
                impact_pct = weight * 100
                factor_explanations.append(f"Menaikkan +{impact_pct:.2f}%")
            else:
                factor_colors.append('#e74c3c')
                impact_pct = abs(weight) * 100
                factor_explanations.append(f"Menurunkan -{impact_pct:.2f}%")
        
        y_pos = np.arange(len(factor_names))
        
        bars = ax_factors.barh(y_pos, factor_values, color=factor_colors, alpha=0.8, 
                              edgecolor='black', linewidth=1.8, height=0.65)
        
        for i, (bar, expl) in enumerate(zip(bars, factor_explanations)):
            width = bar.get_width()
            ax_factors.text(width + 0.002, i, f'  {expl}', 
                          va='center', ha='left', fontsize=11, fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                   alpha=0.9, edgecolor='gray', linewidth=1))
        
        ax_factors.set_yticks(y_pos)
        ax_factors.set_yticklabels(factor_names, fontsize=12, fontweight='bold')
        ax_factors.set_xlabel('Dampak terhadap Probabilitas Prediksi', fontsize=13, fontweight='bold', labelpad=10)
        ax_factors.set_title('Faktor-Faktor Utama yang Mempengaruhi Prediksi', 
                           fontsize=14, fontweight='bold', pad=20)
        ax_factors.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
        ax_factors.set_xlim(0, max(factor_values) * 1.35)
        
        positive_patch = plt.Rectangle((0, 0), 1, 1, fc='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.5)
        negative_patch = plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax_factors.legend([positive_patch, negative_patch], 
                        ['Mendukung kategori ini', 'Mengurangi kepercayaan kategori ini'],
                        loc='lower right', fontsize=12, framealpha=0.95, edgecolor='black')
        
        ax_prob = fig.add_subplot(gs[2, 0])
        
        probabilities = pred_proba[0]
        x_pos = np.arange(len(class_names))
        
        colors = []
        for i, cls in enumerate(class_names):
            if cls == predicted_label:
                colors.append('#e74c3c')
            elif probabilities[i] > 0.3:
                colors.append('#f39c12')
            else:
                colors.append('#95a5a6')
        
        bars = ax_prob.bar(x_pos, probabilities, color=colors, alpha=0.85, 
                          edgecolor='black', linewidth=2.5, width=0.6)
        
        ax_prob.set_xticks(x_pos)
        ax_prob.set_xticklabels(class_names, fontsize=12, fontweight='bold')
        ax_prob.set_ylabel('Probabilitas', fontsize=12, fontweight='bold', labelpad=10)
        ax_prob.set_title('Perbandingan Probabilitas Semua Kategori', fontsize=13, fontweight='bold', pad=15)
        ax_prob.set_ylim(0, 1.15)
        ax_prob.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        ax_prob.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax_prob.text(bar.get_x() + bar.get_width()/2., height + 0.04,
                       f'{prob:.2%}', ha='center', va='bottom', 
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))
        
        ax_data = fig.add_subplot(gs[2, 1])
        ax_data.axis('off')
        
        data_summary = "DATA PEMBELAJARAN ANDA\n\n"
        key_features = [
            ("completion_study_duration", "Durasi Belajar"),
            ("submission_duration", "Durasi Tugas"),
            ("completion_avg_submission_rating", "Rating Rata-rata"),
            ("submission_rating", "Rating Tugas"),
            ("completion_enrolling_times", "Kali Mendaftar"),
        ]
        
        for feat, label in key_features:
            if feat in X_original.columns:
                value = X_original[feat].values[0]
                if feat in ["completion_study_duration", "submission_duration"]:
                    data_summary += f"{label}: {value:.0f} menit\n"
                else:
                    data_summary += f"{label}: {value:.1f}\n"
        
        ax_data.text(0.1, 0.9, data_summary, 
                    ha='left', va='top', fontsize=12, 
                    bbox=dict(boxstyle='round,pad=1.2', facecolor='#e8f4f8', alpha=0.95, 
                             edgecolor='#3498db', linewidth=2.5),
                    fontfamily='monospace', linespacing=1.8)
        
        fig.suptitle('Dashboard Analisis Pembelajaran untuk Siswa', 
                    fontsize=17, fontweight='bold', y=0.97)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        
        return image_base64