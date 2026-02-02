## 데이터가 모델에 들어가기 전 '데이터에 결점이 없는 지' 검사하는 Schema 정의

import pandera as pa
from pandera import Column, Check

# Class 방식(SchemaModel)이 에러가 나므로,
# 호환성이 좋은 Object 방식(DataFrameSchema)으로 변경합니다.
FeatureSchema = pa.DataFrameSchema({
    "unit_nr": Column(int, Check.ge(1), coerce=True),      # 1 이상
    "time_cycles": Column(int, Check.ge(1), coerce=True),  # 1 이상
    "RUL": Column(float, Check.ge(0), coerce=True),        # 0 이상 (음수 불가)
    
    # 핵심 센서 체크
    "sensor_11_ema": Column(float, nullable=False),
    "sensor_11_diff": Column(float, nullable=True),        # NaN 허용 (초반 구간)
}, strict=False) # 정의하지 않은 다른 컬럼들도 허용 (Drop 하지 않음)