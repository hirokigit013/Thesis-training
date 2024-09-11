import cv2
import numpy as np
import argparse
import pytesseract
import concurrent.futures
import time

# 最小サイズの設定（面積）アルファベット"O"などが引っかかってしまうため
MIN_CIRCLE_AREA = 50
MIN_ELLIPSE_AREA = 50

# Tesseract OCRエンジンのパスを設定
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# OCRキャッシュを初期化
ocr_cache = {}

def extract_text_from_shape(image, bounding_box):
    x, y, w, h = bounding_box
    roi = image[y:y + h, x:x + w]
    text = pytesseract.image_to_string(roi, config='--psm 6')
    return text.strip()

def mask_text_regions(image):
    # グレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # テキスト領域を検出するためのOCR実行
    config = '--psm 6'
    d = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    
    # マスク画像を作成 (テキスト領域を白、その他を黒にする)
    mask = np.zeros_like(gray)
    
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # 信頼度が60%以上のテキスト領域を使用
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            mask[y:y+h, x:x+w] = 255  # テキスト領域を白にする
    
    # マスク領域を反転 (テキスト領域が黒、その他が白にする)
    mask = cv2.bitwise_not(mask)
    
    # 元の画像にマスクを適用
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image

# 検出された図形が別の図形に内包されているかどうかをチェックし、内包されている図形を除外
def filter_contained_shapes(shapes):
    sorted_shapes = sorted(shapes, key=lambda x: x['bounding_box'][2] * x['bounding_box'][3], reverse=True)
    filtered_shapes = []
    for shape in sorted_shapes:
        x1, y1, w1, h1 = shape['bounding_box']
        ignore = False
        for other_shape in filtered_shapes:
            x2, y2, w2, h2 = other_shape['bounding_box']
            if x2 <= x1 and y2 <= y1 and (x2 + w2) >= (x1 + w1) and (y2 + h2) >= (y1 + h1):
                ignore = True
                break
        if not ignore:
            filtered_shapes.append(shape)
    return filtered_shapes

# 画像中のラインを検出する関数
def find_lines(image):
    edges = cv2.Canny(image, 100, 150, apertureSize=5)
    edges = cv2.dilate(edges, None, iterations=1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=10)
    return lines

# 指定された座標位置にある形状を見つける関数
def find_shape_at(x, y, shapes):
    point = (int(x), int(y))
    for shape in shapes:
        contour = np.array(shape['contour'], dtype=np.int32)
        distance = cv2.pointPolygonTest(contour, point, False)
        if distance >= 0:
            return shape
    return None

# OCRキャッシュを用いて近くの文字を読み取る関数
def read_text_near_arrow_with_cache(image, x1, y1, x2, y2, margin=30):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    key = (center_x, center_y, margin)
    if key in ocr_cache:
        return ocr_cache[key]
    
    roi = image[max(0, center_y - margin):min(image.shape[0], center_y + margin), 
                max(0, center_x - margin):min(image.shape[1], center_x + margin)]
    text = pytesseract.image_to_string(roi, config='--psm 6')
    ocr_cache[key] = text.strip()
    return ocr_cache[key]

def build_shape_index(shapes):
    shape_index = {}
    for shape_id, shape in enumerate(shapes):
        x, y, w, h = shape['bounding_box']
        for i in range(x, x + w):
            for j in range(y, y + h):
                shape_index[(i, j)] = shape_id
    return shape_index

def find_shape_at(x, y, shape_index, shapes):
    if (x, y) in shape_index:
        shape_id = shape_index[(x, y)]
        return shapes[shape_id]
    return None

def check_arrow_connections(shapes, lines, image):
    connections = {shape_id: {"shape": shape, "connected_arrows": [], "texts": []} for shape_id, shape in enumerate(shapes)}
    shape_index = build_shape_index(shapes)

    if lines is not None:
        def process_line(line):
            x1, y1, x2, y2 = line[0]
            from_shape = find_shape_at(x1, y1, shape_index, shapes)
            to_shape = find_shape_at(x2, y2, shape_index, shapes)

            if from_shape and from_shape == to_shape:
                return None
            
            text = read_text_near_arrow_with_cache(image, x1, y1, x2, y2)
            return from_shape, to_shape, (x1, y1, x2, y2), text
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_line, lines))
        
        for result in results:
            if result is None:
                continue
            from_shape, to_shape, arrow, text = result
            if from_shape and to_shape:
                from_id = shapes.index(from_shape)
                to_id = shapes.index(to_shape)
                connections[from_id]["connected_arrows"].append(arrow)
                connections[to_id]["connected_arrows"].append(arrow)
                connections[from_id]["texts"].append(text)
                connections[to_id]["texts"].append(text)

    return connections

# 楕円かどうかを判定する関数
def is_ellipse(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    aspect_ratio = float(cv2.boundingRect(contour)[2]) / cv2.boundingRect(contour)[3]
    # 小さな楕円を除外
    #return 0.7 <= circularity <= 1.2 and 0.8 <= aspect_ratio <= 1.2 and area >= MIN_ELLIPSE_AREA
    return 0.7 <= circularity <= 1.2 and area >= MIN_ELLIPSE_AREA

# 正円かどうかを判定する関数
def is_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * radius ** 2
    # 小さな円を除外
    return 0.9 <= area / circle_area <= 1.1 and area >= MIN_CIRCLE_AREA

# 角度を計算する関数
def angle_between(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cos_angle))
    return angle

# 角度を調べて菱形かどうかを判定する関数
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

# 画像中の形状を検出する関数
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
        if len(approx) == 4 and not is_ellipse(contour):  # 楕円チェックを追加
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
        
        # OCRを用いて図形の中の文字を読み取る
        text = extract_text_from_shape(image, (x, y, w, h))
        
        shapes.append({'type': shape_type, 'bounding_box': (x, y, w, h), 'contour': approx, 'text': text})

    # 内包されている図形を除外する
    shapes = filter_contained_shapes(shapes)
    
    return shapes
# 折れ線矢印を検出する関数
def find_polyline_arrows(lines):
    polylines = []
    
    if lines is not None:
        visited = set()
        
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

# メイン関数
def main(image_path):
    # 画像ファイルを読み込む
    image = cv2.imread(image_path)

    if image is None:
        print('Failed to load the image. Please check the file path.')
        return

    # 形状を検出
    shapes = detect_shapes(image)

    # 検出した形状の情報をログに出力する
    for shape in shapes:
        x, y, w, h = shape["bounding_box"]
        shape_type = shape["type"]
        text = shape["text"]
        print(f"Shape: {shape_type}, Coordinates: (x: {x}, y: {y}, w: {w}, h: {h}), Text: '{text}'")
        # 輪郭を緑色で描画
        cv2.drawContours(image, [np.array(shape["contour"])], -1, (0, 255, 0), 2)
        
    # グレースケールで画像を変換して線を検出
    lines = find_lines(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    start_time = time.time()
    # 形状とラインの関係をチェック
    connections = check_arrow_connections(shapes, lines, image)
    end_time = time.time()
    print(end_time - start_time)
    """
    # 折れ線矢印を検出
    polyline_arrows = find_polyline_arrows(lines)

    # 重複しない接続情報を出力
    printed_connections = set()
    for shape_id, connection in connections.items():
        shape = connection["shape"]
        shape_type = shape["type"]
        # "unknown" の図形はスキップする
        if shape_type == "unknown":
            continue
        arrows = connection["connected_arrows"]
        texts = connection["texts"]
        for i, arrow in enumerate(arrows):
            x1, y1, x2, y2 = arrow
            from_shape = find_shape_at(x1, y1, shapes)
            to_shape = find_shape_at(x2, y2, shapes)
            if from_shape and to_shape:
                from_id = shapes.index(from_shape)
                to_id = shapes.index(to_shape)
                if (from_id, to_id) not in printed_connections:
                    text = texts[i]
                    print(f"Shape ID {from_id} ({from_shape['type']}) is connected to Shape ID {to_id} ({to_shape['type']}) with arrow: {arrow}, Text: '{text}'")
                    printed_connections.add((from_id, to_id))

    # フローチャートの各ステップを画像に描画
    for shape_id, connection in connections.items():
        shape = connection["shape"]
        shape_type = shape["type"]
        # "unknown" の図形はスキップする
        if shape_type == "unknown":
            continue
        arrows = connection["connected_arrows"]
        x, y, w, h = shape["bounding_box"]
        # 図形を緑色の矩形で囲む
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for arrow in arrows:
            x1, y1, x2, y2 = arrow
            # 矢印の始点と終点を赤い線で描画
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 折れ線矢印をオレンジ色で描画
    for points in polyline_arrows:
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i + 1], (0, 140, 255), 3)
        cv2.arrowedLine(image, points[-2], points[-1], (0, 140, 255), 3, tipLength=0.2)
        # 折れ線矢印をオレンジ色で描画し、始点と終点をターミナルに出力
        print(f"Polyline arrow start point: {points[0]}, end point: {points[-1]}")
"""
    # 画像をウィンドウに表示
    cv2.imshow("Flowchart", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # コマンドライン引数をパース
    parser = argparse.ArgumentParser(description="Detect and organize shapes and lines in a given image.")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    args = parser.parse_args()

    # 指定された画像パスを用いてメイン関数を実行
    main(args.image_path)