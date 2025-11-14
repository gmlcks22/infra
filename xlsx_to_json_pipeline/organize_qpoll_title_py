import os
import json
import glob
import pandas as pd
import re

# --- 1. 경로 설정 ---
# 이 스크립트 파일(create_codebook.py)의 위치
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../infra/embedding_preprocessing
# 상위 'infra' 폴더
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # .../infra

# [입력] 원본 qpoll 엑셀 파일이 있는 폴더
INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    'xlsx_to_json_pipeline',
    'data',
    'Quickpoll'
)
INPUT_XLSX_FILES = glob.glob(os.path.join(INPUT_DIR, 'qpoll*.xlsx'))

# [출력] 모든 질문과 답변을 요약할 단일 파일
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'qpoll_title_organization.json')

def parse_sheet2_for_codebook(file_path):
    """
    하나의 엑셀 파일에서 Sheet2만 읽어, 
    질문 텍스트와 답변 보기 목록을 추출합니다.
    """
    
    # 파일명에서 'qpoll_join_250106' 같은 ID 추출
    base_name = os.path.basename(file_path)
    topic_file_id, _ = os.path.splitext(base_name)
    
    extracted_data = [] # 이 파일에서 추출한 질문 목록
    
    try:
        xlsx = pd.ExcelFile(file_path)
        
        # [핵심] 두 번째 시트(Sheet2)만 파싱
        df_labels = xlsx.parse(xlsx.sheet_names[1], header=None)
        
        row_index = 0
        while row_index < len(df_labels):
            
            # 1. '설문제목' 행 찾기 (보기 ID가 있는 행)
            id_row_value = df_labels.iloc[row_index, 0] # A열
            if pd.isna(id_row_value) or id_row_value.strip() != "설문제목":
                row_index += 1
                continue
            
            # 2. '총참여자수' 위치 찾기 (이전 로직 동일)
            id_row_data_series = df_labels.iloc[row_index, 1:]
            stop_col_pos = None
            for i, item in enumerate(id_row_data_series):
                if pd.notna(item) and str(item).strip() == '총참여자수':
                    stop_col_pos = i 
                    break
            
            # 3. 다음 행으로 이동 (질문 텍스트 + 답변 라벨 행)
            row_index += 1
            if row_index >= len(df_labels):
                break
            
            # 4. [추출 1] A열에서 '설문 제목' (질문 텍스트) 추출
            question_text = df_labels.iloc[row_index, 0]
            if pd.isna(question_text):
                 question_text = "N/A" # A열이 비어있는 경우
            question_text = question_text.strip()
            
            # 5. [추출 2] B열부터 '답변 종류' (라벨) 추출
            label_row_data_series = df_labels.iloc[row_index, 1:]
            
            labels_to_use = None
            if stop_col_pos is not None:
                labels_to_use = label_row_data_series.iloc[:stop_col_pos].values
            else:
                labels_to_use = label_row_data_series.values
            
            # 6. NaN이 아닌 유효한 답변 보기만 리스트로 정리
            answer_options = [
                str(label) for label in labels_to_use if pd.notna(label)
            ]
            
            # 7. 최종 데이터 추가
            extracted_data.append({
                "source_file_id": topic_file_id,
                "question_text": question_text,
                "answer_options": answer_options
            })
            
            # 8. 다음 '설문제목' 블록으로 이동
            row_index += 1
            
    except Exception as e:
        print(f"  > 파일 처리 오류 ({base_name}): {e}")
        
    return extracted_data

# --- 4. 메인 실행 로직 ---

def main():
    if not INPUT_XLSX_FILES:
        print(f"오류: '{INPUT_DIR}' 폴더에서 qpoll*.xlsx 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(INPUT_XLSX_FILES)}개의 엑셀 파일에서 Sheet2를 읽어옵니다...")
    
    # 모든 파일의 질문을 합칠 마스터 리스트
    all_questions_bank = []
    
    for file_path in INPUT_XLSX_FILES:
        print(f"  > 처리 중: {os.path.basename(file_path)}")
        # 각 파일의 Sheet2에서 데이터를 추출하여 마스터 리스트에 추가
        all_questions_bank.extend(parse_sheet2_for_codebook(file_path))

    # --- 최종 파일 저장 ---
    if not all_questions_bank:
        print("\n추출된 질문 데이터가 없습니다.")
        return

    print(f"\n--- 작업 완료 ---")
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_questions_bank, f, ensure_ascii=False, indent=4)
        
        print(f"성공! 총 {len(all_questions_bank)}개의 질문 코드가 '{OUTPUT_FILE}'에 저장되었습니다.")
        
        if all_questions_bank:
            print("\n--- 첫 번째 질문 데이터 예시 ---")
            print(json.dumps(all_questions_bank[0], indent=4, ensure_ascii=False))

    except Exception as e:
        print(f"최종 파일 저장 오류: {e}")

if __name__ == '__main__':
    main()