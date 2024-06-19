import base64
import requests
from datetime import datetime, timedelta
from OpenSSL import crypto
from zeep import Client
from zeep import Client
from zeep.transports import Transport
from requests import Session

wsaa_url = 'https://wsaa.afip.gov.ar/ws/services/LoginCms?wsdl'
wsfe_url = 'https://servicios1.afip.gov.ar/wsfev1/service.asmx?WSDL'


def create_tra(service):
    tra = f"""<?xml version="1.0" encoding="UTF-8" ?>
    <loginTicketRequest version="1.0">
        <header>
            <uniqueId>{int(datetime.now().timestamp())}</uniqueId>
            <generationTime>{(datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%S")}</generationTime>
            <expirationTime>{(datetime.now() + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%S")}</expirationTime>
        </header>
        <service>{service}</service>
    </loginTicketRequest>"""
    return tra

def sign_tra(tra, private_key_path, cert_path):
    pkey = crypto.load_privatekey(crypto.FILETYPE_PEM, open(private_key_path).read())
    cert = crypto.load_certificate(crypto.FILETYPE_PEM, open(cert_path).read())
    pkcs7 = crypto.sign(cert, tra, 'sha256')
    return base64.b64encode(pkcs7).decode('utf-8')

def request_ta(tra, private_key_path, cert_path, wsaa_url):
    tra_signed = sign_tra(tra, private_key_path, cert_path)
    login_cms = f"""<loginCmsRequest>
        <loginTicketRequest>{tra_signed}</loginTicketRequest>
    </loginCmsRequest>"""
    response = requests.post(wsaa_url, data=login_cms, headers={'Content-Type': 'text/xml'})
    return response.content

private_key_path = 'private_key.key'
cert_path = 'certificate.crt'
wsaa_url = 'https://wsaa.afip.gov.ar/ws/services/LoginCms?wsdl'

tra = create_tra('wsfe')
ta_response = request_ta(tra, private_key_path, cert_path, wsaa_url)
print(ta_response)

import pdb 
pdb.set_trace()
#---------------------------------------------
# URLs y paths
wsfe_url = 'https://servicios1.afip.gov.ar/wsfev1/service.asmx?WSDL'
cuit = 'YOUR_CUIT'
token = 'YOUR_TOKEN'
sign = 'YOUR_SIGN'

# Crear cliente
client = Client(wsdl=wsfe_url)

# Crear solicitud para obtener últimos comprobantes
request = {
    'Auth': {
        'Token': token,
        'Sign': sign,
        'Cuit': cuit
    },
    'PtoVta': 1,  # Punto de venta
    'CbteTipo': 1  # Tipo de comprobante, e.g. Factura A
}

response = client.service.FECompUltimoAutorizado(request)
print(response)
