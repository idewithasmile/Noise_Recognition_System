#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <ArduinoJson.h>

const char* apSSID = "ESP32-Audio";
const char* apPassword = "12345678";

AsyncWebServer server(80);
WebSocketsClient webSocket;

String ssid = "";
String password = "";
String serverIP = "";

bool isAPMode = true;
bool streaming = false;
bool webSocketInitialized = false;

#define I2S_DATA_IN_PIN 13
#define I2S_SCK_PIN 2
#define I2S_WS_PIN 15
#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024
int16_t audioBuffer[BUFFER_SIZE];

unsigned long lastSendTime = 0;
const int sendInterval = 20;
unsigned long lastHeartbeatTime = 0;
const int heartbeatInterval = 30000;

// === I2S cấu hình ===
void setupI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK_PIN,
    .ws_io_num = I2S_WS_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_DATA_IN_PIN
  };

  // Uninstall driver first if it was installed before
  i2s_driver_uninstall(I2S_NUM_0);
  delay(100);
  
  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_start(I2S_NUM_0);
}

// === WebSocket event handler ===
void webSocketEvent(WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.println("[WS] Đã kết nối tới server");
      break;

    case WStype_DISCONNECTED:
      Serial.println("[WS] Mất kết nối WebSocket");
      streaming = false;
      break;

    case WStype_TEXT: {  // Thêm dấu ngoặc nhọn để tạo một scope mới
      Serial.printf("[WS] Nhận: %s\n", payload);
      
      // Phân tích JSON
      DynamicJsonDocument doc(256);
      DeserializationError error = deserializeJson(doc, payload);
      
      if (!error && doc.containsKey("command")) {
        String command = doc["command"];
        Serial.printf("[WS] Nhận lệnh: %s\n", command.c_str());
        
        if (command == "start_stream") {
          streaming = true;
          Serial.println(">> BẮT ĐẦU STREAM AUDIO");
        } else if (command == "stop_stream") {
          streaming = false;
          Serial.println(">> DỪNG STREAM AUDIO");
        }
      } else {
        // Thử kiểm tra chuỗi thông thường
        if (strstr((char*)payload, "start_stream") != NULL) {
          streaming = true;
          Serial.println(">> BẮT ĐẦU STREAM AUDIO");
        } else if (strstr((char*)payload, "stop_stream") != NULL) {
          streaming = false;
          Serial.println(">> DỪNG STREAM AUDIO");
        }
      }
      break;
    }  // Đóng scope
      
    default:
      break;
  }
}

// === Gửi heartbeat định kỳ ===
void sendHeartbeat() {
  if (webSocketInitialized && webSocket.isConnected()) {
    DynamicJsonDocument doc(128);
    doc["type"] = "heartbeat";
    doc["uptime"] = millis() / 1000;
    String out;
    serializeJson(doc, out);
    webSocket.sendTXT(out);
  }
}

// === Setup Access Point ===
void setupAccessPoint() {
  // Đảm bảo tắt WiFi trước
  WiFi.disconnect(true);
  delay(100);

  // Bật chế độ AP
  WiFi.mode(WIFI_MODE_AP);
  isAPMode = true;

  // Tạo Access Point
  WiFi.softAP(apSSID, apPassword);

  Serial.println("== Access Point đã khởi động ==");
  Serial.print("SSID: "); Serial.println(apSSID);
  Serial.print("IP: "); Serial.println(WiFi.softAPIP());

  // Khởi động lại server
  setupWebServer();  // Đảm bảo gọi lại hàm này để cấu hình route
}

void disableAP() {
  if (WiFi.softAPdisconnect(true)) {
    Serial.println("AP đã tắt thành công");
    isAPMode = false;
  } else {
    Serial.println("Không thể tắt AP");
  }
}

void safeRestartNetwork() {
  Serial.println("[Network] Đang khởi động lại mạng...");

  // Đặt cờ hiệu cho biết WebSocket không còn khởi tạo
  webSocketInitialized = false;
  streaming = false;

  // Dừng các dịch vụ hiện tại
  webSocket.disconnect();
  server.end();

  // Ngắt kết nối phù hợp với chế độ hiện tại
  if (isAPMode) {
    WiFi.softAPdisconnect(true);  // Tắt AP
  } else {
    WiFi.disconnect(true);        // Ngắt WiFi STA
  }

  delay(1000);  // Đợi ổn định

  // Khởi động lại đúng chế độ
  if (isAPMode) {
    Serial.println("[Network] Bật lại Access Point...");
    setupAccessPoint();
  } else {
    Serial.println("[Network] Kết nối lại WiFi & WebSocket...");
    connectToWiFiAndServer();
  }
}

void connectToWiFiAndServer() {
  Serial.println("[WiFi] Đang kết nối đến SSID: " + ssid);

  // Đặt cờ hiệu cho biết WebSocket không còn khởi tạo
  webSocketInitialized = false;
  streaming = false;

  // Dừng WebSocket và server nếu đang chạy
  webSocket.disconnect();
  server.end();

  // Đặt chế độ về STA-only
  WiFi.mode(WIFI_MODE_STA);
  WiFi.disconnect(true);  // Xóa cấu hình cũ nếu có
  delay(200);             // Cho ESP32 thời gian chuyển trạng thái

  WiFi.begin(ssid.c_str(), password.c_str());

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    isAPMode = false;
    Serial.println("\n[WiFi] Đã kết nối!");
    Serial.print("[WiFi] IP: "); Serial.println(WiFi.localIP());

    // Kết nối WebSocket
    webSocket.begin(serverIP.c_str(), 5000, "/");
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    webSocketInitialized = true;  // Đánh dấu rằng WebSocket đã được khởi tạo

    // Khởi động lại web server nếu cần
    setupWebServer();
  } else {
    Serial.println("\n[WiFi] Kết nối thất bại!");

    // Quay lại chế độ AP
    WiFi.disconnect(true);
    delay(100);
    WiFi.mode(WIFI_MODE_AP);
    setupAccessPoint();
  }
}

// === Giao diện cấu hình WiFi và Server IP ===
void setupWebServer() {
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32 Audio Setup</title>
  <style>
    body {
      font-family: sans-serif;
      background: #f4f4f4;
      padding: 20px;
      margin: 0;
    }
    .container {
      max-width: 400px;
      margin: auto;
      background: #fff;
      padding: 20px;
      border-radius: 12px;
      box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    h2 {
      text-align: center;
      color: #333;
    }
    input, button {
      width: 100%;
      padding: 12px;
      margin: 10px 0;
      font-size: 16px;
      border-radius: 6px;
      border: 1px solid #ccc;
    }
    button {
      background-color: #28a745;
      color: white;
      border: none;
      cursor: pointer;
    }
    button:hover {
      background-color: #218838;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>ESP32 Audio Setup</h2>
    <form action="/config" method="POST">
      <label for="ssid">Wi-Fi SSID:</label>
      <input name="ssid" id="ssid" required>

      <label for="pass">Mật khẩu:</label>
      <input name="pass" id="pass" type="password" required>

      <label for="ip">IP máy chủ WebSocket:</label>
      <input name="ip" id="ip" type="text" placeholder="VD: 192.168.4.2" required>

      <button type="submit">Kết nối</button>
    </form>
  </div>
</body>
</html>
)rawliteral";

    request->send(200, "text/html; charset=utf-8", html);
  });

  server.on("/config", HTTP_POST, [](AsyncWebServerRequest *request){
    if (request->hasParam("ssid", true) &&
        request->hasParam("pass", true) &&
        request->hasParam("ip", true)) {
      
      ssid = request->getParam("ssid", true)->value();
      password = request->getParam("pass", true)->value();
      serverIP = request->getParam("ip", true)->value();

      request->send(200, "text/plain", "Đang kết nối tới WiFi & WebSocket server...");
      
      // Use a short delay to allow the HTTP response to be sent before changing network settings
      // This prevents a crash if we try to send a response while changing network mode
      xTaskCreate([](void *parameter) {
        delay(1000);
        connectToWiFiAndServer();
        vTaskDelete(NULL);
      }, "connect_task", 4096, NULL, 1, NULL);
    } else {
      request->send(400, "text/plain", "Thiếu thông tin");
    }
  });

  // Add a status endpoint
  server.on("/status", HTTP_GET, [](AsyncWebServerRequest *request){
    String status = "Mode: ";
    status += isAPMode ? "AP" : "STA";
    status += "\nWSconnected: ";
    status += (webSocketInitialized && webSocket.isConnected()) ? "Yes" : "No";
    status += "\nIP: ";
    status += isAPMode ? WiFi.softAPIP().toString() : WiFi.localIP().toString();
    
    request->send(200, "text/plain", status);
  });

  server.begin();
}

// === Hàm setup chính ===
void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n=== ESP32 Audio Setup Starting ===");

  WiFi.onEvent([](WiFiEvent_t event, WiFiEventInfo_t info) {
    Serial.printf("[WiFi-EVENT] %d\n", event);
  });
  
  setupI2S();
  setupAccessPoint(); // Bắt đầu với AP mode
}

// === Vòng lặp chính ===
void loop() {
  if (!isAPMode && webSocketInitialized) {
    webSocket.loop();
  }

  // Debug trạng thái mỗi 5 giây
  static unsigned long lastStatusTime = 0;
  if (millis() - lastStatusTime > 5000) {
    lastStatusTime = millis();
    Serial.print("Status: WebSocket connected=");
    Serial.print(webSocket.isConnected());
    Serial.print(", streaming=");
    Serial.println(streaming);
  }

  // Kiểm tra kết nối định kỳ
  static unsigned long lastCheck = 0;
  if (millis() - lastCheck > 30000) {
    lastCheck = millis();
    if (!isAPMode && WiFi.status() != WL_CONNECTED) {
      Serial.println("Mất kết nối WiFi, khôi phục AP...");
      safeRestartNetwork();
    }
  }

  // Only try to read and send audio if all conditions are met
  if (!isAPMode && webSocketInitialized && webSocket.isConnected() && streaming) {
    size_t bytesRead = 0;
    esp_err_t result = i2s_read(I2S_NUM_0, audioBuffer, sizeof(audioBuffer), &bytesRead, 10);
    
    if (result == ESP_OK && bytesRead > 0) {
      if (millis() - lastSendTime > sendInterval) {
        webSocket.sendBIN((uint8_t*)audioBuffer, bytesRead);
        lastSendTime = millis();
      }
    }
  }

  // Only send heartbeat if properly connected
  if (!isAPMode && webSocketInitialized && webSocket.isConnected() && millis() - lastHeartbeatTime > heartbeatInterval) {
    lastHeartbeatTime = millis();
    sendHeartbeat();
  }

  // Add small delay to prevent CPU hogging
  delay(1);
}