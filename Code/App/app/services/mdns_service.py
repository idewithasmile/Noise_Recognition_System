from zeroconf import ServiceInfo, Zeroconf
import socket

class MDNSService:
    """Module quản lý mDNS service discovery"""

    def __init__(self, config):
        self.config = config
        self.logger = config.get_logger('mdns')
        self.zeroconf = Zeroconf()
        self.services = {}  # Lưu lại các service đã đăng ký
        self.logger.info("Đã khởi tạo mDNS service")

    def start(self, service_name="MyServer", service_type="_http._tcp.local.", port=8000):
        """Khởi động mDNS service"""
        try:
            hostname = socket.gethostname()
            local_ip = socket.gethostbyname(hostname)

            # print("[mDNS TEST] Hostname:", hostname)
            # print("[mDNS TEST] Local IP:", local_ip)

            desc = {'path': '/'}
            service_type_full = service_type if service_type.endswith(".") else service_type + "."

            info = ServiceInfo(
                service_type_full,
                f"{service_name}.{service_type_full}",
                addresses=[socket.inet_aton(local_ip)],
                port=port,
                properties=desc,
                server=f"{service_name}.local."  # <-- Đăng ký hostname ESP32 có thể tra
            )

            self.zeroconf.register_service(info)
            self.services[service_name] = info  # lưu lại để gỡ sau này

            self.logger.info(f"Đăng ký mDNS service: {service_name}.{service_type} ({local_ip}:{port})")
            return self

        except Exception as e:
            self.logger.warning(f"Lỗi khi đăng ký mDNS: {e}")
            return None

    def stop(self):
        """Dừng mDNS service"""
        if self.zeroconf:
            self.logger.info("Dừng mDNS service")
            for name, info in self.services.items():
                try:
                    self.zeroconf.unregister_service(info)
                    self.logger.info(f"Đã hủy mDNS service: {name}")
                except Exception as e:
                    self.logger.warning(f"Lỗi khi hủy mDNS service {name}: {e}")
            self.zeroconf.close()
