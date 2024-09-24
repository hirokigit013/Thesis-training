import cv2  # OpenCVライブラリをインポート
import numpy as np  # NumPyライブラリをインポート
import argparse  # コマンドライン引数を解析するためのライブラリをインポート
import pytesseract  # Tesseract OCRエンジンをインポート
import concurrent.futures  # 並列処理をサポートするライブラリをインポート
import time  # 時間測定を行うためのライブラリをインポート
from difflib import SequenceMatcher #類似判定

MIN_CIRCLE_AREA = 50  # 最小円領域の定数を定義
MIN_ELLIPSE_AREA = 50  # 最小楕円領域の定数を定義
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Tesseract OCRエンジンのパスを設定
ocr_cache = {}  # OCRキャッシュを初期化

# 図形から文字を抜出
def extract_text_from_shape(image, bounding_box, shape_type):
    x, y, w, h = bounding_box  # バウンディングボックスの座標と寸法を取得
    
    # 縮小するマージンを定義（例えば10ピクセル）
    margin = 15
    
    # マージンを考慮して関心領域を縮小
    x_new = x + margin
    y_new = y + margin
    w_new = max(w - 2 * margin, 1)  # 幅は少なくとも1ピクセルにする
    h_new = max(h - 2 * margin, 1)  # 高さも同様

    if shape_type == 'rectangle':
        # 長方形の場合はそのままの大きさを使用
        roi = image[y:y + h, x:x + w]
    else:
        roi = image[y_new:y_new + h_new, x_new:x_new + w_new]  # 関心領域を抽出
    text = pytesseract.image_to_string(roi, config='--psm 6')  # OCRを使用して文字列を抽出
    return text.strip()  # 抽出したテキストをトリムして返す

#図形の中に図形が含まれている場合、含まれている画像は抽出しないようにする
def filter_contained_shapes(shapes):
    sorted_shapes = sorted(shapes, key=lambda x: x['bounding_box'][2] * x['bounding_box'][3], reverse=True) # 一度にすべての図形の領域を計算し、大きい順にソートする。
    filtered_shapes = [] #フィルタリング後に図形を保存するリストを初期化
    for shape in sorted_shapes: # 抽出されたそれぞれの図形をチェックする。
        x1, y1, w1, h1 = shape['bounding_box'] # 各図形のバウンディングボックス（x, y, 幅, 高さ）を取得する。
        ignore = False # 現在の図形が他の図形に包含されているかどうかのフラグを初期化。
        for other_shape in filtered_shapes: #画像分ループ
            x2, y2, w2, h2 = other_shape['bounding_box'] # 比較対象の図形のバウンディングボックスを取得する。
            if x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1): # 現在の図形が他の図形に包含されているかをチェック。
                ignore = True #フラグON
                break
        if not ignore: #フラグがONされなかった場合、フィルタリング後のリストに追加する。
            filtered_shapes.append(shape) 
    return filtered_shapes #フィルタリング後のリストを返す

# 検出された直線を見つける
def find_lines(image):
    edges = cv2.Canny(image, 100, 150, apertureSize=5) #画像からエッジ（輪郭）を検出するためにCannyエッジ検出を実行する。引数は閾値。
    edges = cv2.dilate(edges, None, iterations=1) # エッジを膨張させて線を太くする
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=10) # エッジ画像から直線を検出するためにHough変換を実行する。引数はパラメータ設定。
    return lines #検出された直線を返す

# 図形の検出
def find_shape_at(x, y, shapes): 
    point = (int(x), int(y)) # 指定された座標（x, y）を格納する。
    for shape in shapes: #図形毎に
        contour = np.array(shape['contour'], dtype=np.int32) # 図形の輪郭をNumPy配列として取得する。
        distance = cv2.pointPolygonTest(contour, point, False)
        if distance >= 0: # ポイントが輪郭内にある場合、その図形を返す。
            return shape
    return None # ポイントがどの図形の内側にもない場合、Noneを返す。

def determine_yes_no(text):
    # 大文字・小文字を区別しないで扱う
    text = text.strip().upper()

    # 文字列の類似度を計算する
    yes_similarity = SequenceMatcher(None, text, "YES").ratio()
    no_similarity = SequenceMatcher(None, text, "NO").ratio()

    # 類似度が高い方を返す
    if yes_similarity > no_similarity:
        return "YES"
    else:
        return "NO"

#矢印の近くにある文字を抽出する
def read_text_near_arrow_with_cache(image, x1, y1, x2, y2, margin=10):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2 #矢印の中心座標を抽出
    key = (center_x, center_y, margin) # 中心座標とマージンをキーとしてキャッシュキーを作成する。
    if key in ocr_cache: # キャッシュにキーが存在する場合、その値（OCR結果テキスト）を返す。
        return ocr_cache[key]
    
    roi = image[max(0, center_y - margin):min(image.shape[0], center_y + margin), 
                max(0, center_x - margin):min(image.shape[1], center_x + margin)]
    text = pytesseract.image_to_string(roi, config='--psm 6').strip().upper()

    # OCR結果が「YES」か「NO」でない場合、最も近いものを推測
    if text not in ["YES", "NO"]:
        text = determine_yes_no(text)

    ocr_cache[key] = text
    return ocr_cache[key]# キャッシュの値を返す。

# 矢印の接続情報を記載
def check_arrow_connections(shapes, lines, image):
    connections = {shape_id: {"shape": shape, "connected_arrows": [], "texts": []} for shape_id, shape in enumerate(shapes)} # 各図形に対して接続情報（connected_arrows, texts）を初期化した辞書を作成。

    if lines is not None: #直線が存在する場合
        def process_line(line):
            x1, y1, x2, y2 = line[0] # 直線の始点と終点の座標を取得
            from_shape = find_shape_at(x1, y1, shapes) # 始点が含まれる図形を特定
            to_shape = find_shape_at(x2, y2, shapes) # 終点が含まれる図形を特定

            if from_shape is to_shape: #もし始点=終点の場合
                return None
            
            text = read_text_near_arrow_with_cache(image, x1, y1, x2, y2) #矢印の近くにある
            return (from_shape, to_shape, (x1, y1, x2, y2), text) # 始点、終点、矢印座標、テキストを返す
        
        # スレッドプールを使って並列で各直線の処理を実行
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_line, lines))
        
        # 並列処理の結果を処理して接続情報に追加
        for result in filter(None,results):
            from_shape, to_shape, arrow, text = result
            if from_shape and to_shape:
                # 始点図形と終点図形のインデックスを取得
                from_id = shapes.index(from_shape)
                to_id = shapes.index(to_shape)
                # 始点と終点の図形情報に矢印とテキストを追加
                connections[from_id]["connected_arrows"].append(arrow)
                connections[to_id]["connected_arrows"].append(arrow)
                connections[from_id]["texts"].append(text)
                connections[to_id]["texts"].append(text)

    return connections #接続情報を返却


# 直線間の角度を測定
def angle_between(p1, p2, p3): 
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# 楕円かを判断
def is_ellipse(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    aspect_ratio = float(cv2.boundingRect(contour)[2]) / cv2.boundingRect(contour)[3]
    return 0.7 <= circularity <= 1.2 and area >= MIN_ELLIPSE_AREA

# 正円かを判断
def is_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * radius ** 2
    return 0.9 <= area / circle_area <= 1.1 and area >= MIN_CIRCLE_AREA

# 菱形かを判断
def is_rhombus(approx):
    if len(approx) != 4:
        return False

    d1 = np.linalg.norm(approx[0][0] - approx[1][0])
    d2 = np.linalg.norm(approx[1][0] - approx[2][0])
    d3 = np.linalg.norm(approx[2][0] - approx[3][0])
    d4 = np.linalg.norm(approx[3][0] - approx[0][0])

    if abs(d1 - d3) < 0.1 * d1 and abs(d2 - d4) < 0.1 * d2:
        angles = []
        for i in range(4):
            p1 = tuple(approx[i % 4][0])
            p2 = tuple(approx[(i + 1) % 4][0])
            p3 = tuple(approx[(i + 2) % 4][0])
            angles.append(angle_between(p1, p2, p3))

        if any(angle < 80 or angle > 100 for angle in angles):
            return True

    return False

#図形の種類を判別
def detect_shapes(image):
    shapes = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        if x == 0 or y == 0:
            continue
        if len(approx) == 4 and not is_ellipse(contour):
            if is_rhombus(approx):
                shape_type = 'rhombus'
            else:
                shape_type = 'rectangle'
        elif is_circle(contour):
            shape_type = 'circle'
        elif is_ellipse(contour):
            shape_type = 'ellipse'
        else:
            shape_type = 'unknown'
            continue

        text = extract_text_from_shape(image, (x, y, w, h),shape_type)
        shapes.append({'type': shape_type, 'bounding_box': (x, y, w, h), 'contour': approx, 'text': text})

    shapes = filter_contained_shapes(shapes)
    return shapes

# 折れ線矢印を判別
def find_polyline_arrows(lines):
    polylines = []

    if lines is not None:
        visited = set()

        # 与えられた3点が角（直角）を形成しているかどうか判定する補助関数
        def is_corner(p1, p2, p3):
            angle = abs(angle_between(p1, p2, p3))
            return 90 <= angle <= 100

        for i in range(len(lines)):
            if i in visited:
                continue
            current_line = lines[i][0]
            x1, y1, x2, y2 = current_line
            polyline = [(x1, y1), (x2, y2)]
            visited.add(i)
            extended = True

            while extended:
                extended = False
                for j in range(len(lines)):
                    if j in visited:
                        continue
                    next_line = lines[j][0]
                    nx1, ny1, nx2, ny2 = next_line
                    if (polyline[-1] == (nx1, ny1) and is_corner(polyline[-2], polyline[-1], (nx2, ny2))):
                        polyline.append((nx2, ny2))
                        visited.add(j)
                        extended = True
                    elif (polyline[-1] == (nx2, ny2) and is_corner(polyline[-2], polyline[-1], (nx1, ny1))):
                        polyline.append((nx1, ny1))
                        visited.add(j)
                        extended = True

            if len(polyline) >= 4:
                polylines.append(polyline)

    return polylines

def main(image_path, run_connection_check):
    image = cv2.imread(image_path)  # 画像のパス

    if image is None:  # 画像がない場合
        print('Failed to load the image. Please check the file path.')
        return

    shapes = detect_shapes(image)  # 画像から図形を抽出

    # y座標でソート
    sorted_shapes = sorted(shapes, key=lambda shape: shape['bounding_box'][1])
    
    # 図形IDを追加して表示
    for shape_id, shape in enumerate(sorted_shapes):
        x, y, w, h = shape["bounding_box"]
        shape_type = shape["type"]
        text = shape["text"]
        print(f"ID: {shape_id}, Shape: {shape_type}, Coordinates: (x: {x}, y: {y}, w: {w}, h: {h}), Text: '{text}'")
        cv2.drawContours(image, [np.array(shape["contour"])], -1, (0, 255, 0), 2)

    lines = find_lines(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  # 画像から直線を抽出
    start_time = time.time()

    # 形状とラインの関係をチェック（フラグに基づいて実行）
    if run_connection_check:
        connections = check_arrow_connections(sorted_shapes, lines, image)
    else:
        connections = {shape_id: {"shape": shape, "connected_arrows": [], "texts": []} for shape_id, shape in enumerate(sorted_shapes)}

    printed_connections = set()
    for shape_id, connection in connections.items():
        shape = connection["shape"]
        shape_type = shape["type"]
        if shape_type == "unknown":
            continue
        arrows = connection["connected_arrows"]
        texts = connection["texts"]
        for i, arrow in enumerate(arrows):
            x1, y1, x2, y2 = arrow
            from_shape = find_shape_at(x1, y1, sorted_shapes)
            to_shape = find_shape_at(x2, y2, sorted_shapes)
            if from_shape and to_shape:
                from_id = sorted_shapes.index(from_shape)
                to_id = sorted_shapes.index(to_shape)
                if from_id < to_id and (from_id, to_id) not in printed_connections:
                    text = texts[i]
                    from_text = from_shape["text"]
                    to_text = to_shape["text"]
                    if from_shape['type'] == 'rhombus':
                        print(f"{from_text} -> {text} -> {to_text}")
                    else:
                        print(f"{from_text} -> {to_text}")
                    print(f"Shape ID {from_id} ({from_shape['type']}) is connected to Shape ID {to_id} ({to_shape['type']})")
                    printed_connections.add((from_id, to_id))

    for shape_id, connection in connections.items():
        shape = connection["shape"]
        shape_type = shape["type"]
        if shape_type == "unknown":
            continue
        arrows = connection["connected_arrows"]
        x, y, w, h = shape["bounding_box"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for arrow in arrows:
            x1, y1, x2, y2 = arrow
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    end_time = time.time()
    print("Processing time:", end_time - start_time)
    cv2.imshow("Flowchart", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
    for points in polyline_arrows:
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], (0, 140, 255), 3)
        cv2.arrowedLine(image, points[-2], points[-1], (0, 140, 255), 3, tipLength=0.2)
        print(f"Polyline arrow start point: {points[0]}, end point: {points[-1]}")
    
    end_time = time.time()
    print("Processing time:", end_time - start_time)
    cv2.imshow("Flowchart", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and organize shapes and lines in a given image.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--skip", action="store_true", help="Skip the arrow connections check")
    args = parser.parse_args()

    main(args.image_path, not args.skip)