import json
import os

# --- 1. 경로 설정 ---
# 현재 스크립트 위치: .../cleanse_w_categorization/update_welcome_w_cat.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트 (infra): 한 단계 상위 폴더
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 매핑 파일들이 있는 폴더
MAPS_DIR = os.path.join(SCRIPT_DIR, 'cat_maps')

# 입력 데이터 파일 경로 (xlsx_to_json_pipeline/welcome_json_output/welcome_data.json)
INPUT_FILE_PATH = os.path.join(
    PROJECT_ROOT, 
    'xlsx_to_json_pipeline', 
    'welcome_json_output', 
    'welcome_data.json'
)

OUTPUT_FILE_PATH = os.path.join(
    PROJECT_ROOT, 
    'cleanse_w_categorization',
    'welcome_data_cleansed.json'
)

# --- 2. 설정: 어떤 매핑 파일을 어떤 필드에 적용할지 정의 ---
# (파일명, 대상 필드명)
MAPPING_CONFIG = [
    ("cat_job_title_raw.json", "job_title_raw"),
    ("cat_car_manufacturer_raw.json", "car_manufacturer_raw"),
    ("cat_phone_brand_raw.json", "phone_brand_raw"),
    ("cat_phone_model_raw.json", "phone_model_raw"),
    # 'dont_cat'이라고 되어있지만 요청에 따라 포함합니다. 필요 없으면 주석 처리하세요.
    #("dont_cat__car_model_raw.json", "car_model_raw") 
]

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  > 파일 로드 실패 ({os.path.basename(path)}): {e}")
        return None

def create_reverse_mapping(category_data):
    """
    카테고리 JSON을 { "개별값": "카테고리명" } 형태의 딕셔너리로 변환
    """
    mapping = {}
    for category, items in category_data.items():
        for item in items:
            clean_item = item.strip()
            mapping[clean_item] = category
    return mapping

def main():
    # 1. 원본 데이터 로드
    print(f"원본 데이터 로드 중...: {INPUT_FILE_PATH}")
    welcome_data = load_json(INPUT_FILE_PATH)
    if not welcome_data:
        return

    total_changes = 0

    # 2. 설정된 매핑 파일들을 순회하며 적용
    for map_filename, target_field in MAPPING_CONFIG:
        map_path = os.path.join(MAPS_DIR, map_filename)
        
        print(f"\n--- 매핑 적용 중: {map_filename} -> {target_field} ---")
        
        # 매핑 파일 로드
        category_data = load_json(map_path)
        if not category_data:
            continue
            
        # 검색용 역방향 맵 생성
        reverse_map = create_reverse_mapping(category_data)
        field_changes = 0

        # 데이터 순회 및 업데이트
        for record in welcome_data:
            original_value = record.get(target_field)
            
            # 값이 존재하고 문자열인 경우에만 처리
            if original_value and isinstance(original_value, str):
                clean_value = original_value.strip()
                
                # 매핑 맵에 값이 있으면 카테고리로 교체
                if clean_value in reverse_map:
                    new_category = reverse_map[clean_value]
                    
                    # 값이 실제로 바뀔 때만 업데이트
                    if original_value != new_category:
                        record[target_field] = new_category
                        field_changes += 1
        
        total_changes += field_changes
        print(f"  > {field_changes}개의 데이터가 변경되었습니다.")

    # 3. 결과 저장
    print(f"\n모든 작업 완료! 총 {total_changes}개의 값이 변경되었습니다.")
    try:
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(welcome_data, f, ensure_ascii=False, indent=4)
        print(f"결과가 저장되었습니다: {OUTPUT_FILE_PATH}")
    except Exception as e:
        print(f"파일 저장 실패: {e}")

if __name__ == "__main__":
    main()