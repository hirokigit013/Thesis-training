import cv2
import numpy as np
import argparse
import pytesseract

# Tesseract OCRエンジンのパスを設定
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# 画像中のラインを検出する関数
def find_lines(image):
    # Cannyエッジ検出を使用してエッジを検出
    edges = cv2.Canny(image, 100, 150, apertureSize=5)  # 閾値を調整
    # 収縮処理を追加して画像を縮小(iterations=繰り返し回数)
    # edges = cv2.erode(edges, None, iterations=1)
    # 膨張処理を追加して画像を膨張(iterations=繰り返し回数)
    edges = cv2.dilate(edges, None, iterations=1)
    # Hough変換を使用して画像中の直線を検出
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=10)  # パラメータを調整
    return lines

# 指定された座標位置にある形状を見つける関数
def find_shape_at(x, y, shapes):
    point = (int(x), int(y))  # 座標を整数に変換
    for shape in shapes:
        contour = np.array(shape['contour'], dtype=np.int32)  # ContourをNumPy配列に変換
        distance = cv2.pointPolygonTest(contour, point, False) # 図形の中に座標が含まれているか
        if distance >= 0:
            return shape
    return None

# OCRを用いて近くの文字を読み取る関数
def read_text_near_arrow(image, x1, y1, x2, y2, margin=50):
    min_x, max_x = sorted([int(x1), int(x2)])
    min_y, max_y = sorted([int(y1), int(y2)])
    
    roi = image[max(0, min_y-margin):min(image.shape[0], max_y+margin), 
                max(0, min_x-margin):min(image.shape[1], max_x+margin)]
    text = pytesseract.image_to_string(roi, config='--psm 6')
    return text.strip()

# 矢印が特定の図形に接続されているかどうかを確認する関数
def check_arrow_connections(shapes, lines, image):
    connections = {shape_id: {"shape": shape, "connected_arrows": [], "texts": []} for shape_id, shape in enumerate(shapes)}

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                from_shape = find_shape_at(x1, y1, shapes)  # 始点の座標
                to_shape = find_shape_at(x2, y2, shapes)  # 終点の座標

                # 矢印が同じ図形の中に収まっている場合、無視
                if from_shape and from_shape == to_shape:
                    continue
                
                # OCRを用いて矢印の近くの文字を読み取る
                text = read_text_near_arrow(image, x1, y1, x2, y2)
                print('aaa')
                if from_shape and to_shape:
                    from_id = shapes.index(from_shape)
                    to_id = shapes.index(to_shape)
                    connections[from_id]["connected_arrows"].append((x1, y1, x2, y2))
                    connections[to_id]["connected_arrows"].append((x1, y1, x2, y2))
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
    return 0.7 <= circularity <= 1.2

# 正円かどうかを判定する関数
def is_circle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * radius ** 2
    return 0.9 <= area / circle_area <= 1.1

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
    # 頂点数確認
    if len(approx) != 4:
        return False

    # 各辺の長さを確認
    d1 = np.linalg.norm(approx[0][0] - approx[1][0])
    d2 = np.linalg.norm(approx[1][0] - approx[2][0])
    d3 = np.linalg.norm(approx[2][0] - approx[3][0])
    d4 = np.linalg.norm(approx[3][0] - approx[0][0])

    # 周辺のほぼ等しい長さを確認
    if abs(d1 - d3) < 0.1 * d1 and abs(d2 - d4) < 0.1 * d2:
        # 確認
        angles = []
        for i in range(4):
            p1 = tuple(approx[i % 4][0])
            p2 = tuple(approx[(i + 1) % 4][0])
            p3 = tuple(approx[(i + 2) % 4][0])
            angles.append(angle_between(p1, p2, p3))

        # 内部角度が約90度なら菱形ではないので false
        if all(80 <= angle <= 100 for angle in angles):
            return False
        return True

    return False

# 画像中の形状を検出する関数 (続き)
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
        if len(approx) == 4:
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
        shapes.append({'type': shape_type, 'bounding_box': (x, y, w, h), 'contour': approx})

    return shapes

# 折れ線矢印を検出する関数
def find_polyline_arrows(lines):
    polylines = []
    
    if lines is not None:
        
        visited = set()
        def is_corner(p1, p2, p3):
            angle = abs(angle_between(p1, p2, p3))
            return 80 <= angle <= 100
        
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
        print(f"Shape: {shape_type}, Coordinates: (x: {x}, y: {y}, w: {w}, h: {h})")
        # 輪郭を緑色で描画
        cv2.drawContours(image, [np.array(shape["contour"])], -1, (0, 255, 0), 2)

    # グレースケールで画像を変換して線を検出
    lines = find_lines(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    # 形状とラインの関係をチェック
    connections = check_arrow_connections(shapes, lines, image)

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