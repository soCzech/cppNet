from http.server import BaseHTTPRequestHandler,HTTPServer
import os
import sys
import argparse

LOGDIR = ""
PORT_NUMBER = 8080

class myHandler(BaseHTTPRequestHandler):
		
	def do_GET(self):
		if self.path=="/":
			self.path="/index.html"

		try:
			sendReply = False
			if self.path.endswith(".html"):
				mimetype='text/html'
				sendReply = True
			
			if self.path.endswith(".js"):
				mimetype='application/javascript'
				sendReply = True
			
			if self.path.endswith(".css"):
				mimetype='text/css'
				sendReply = True
			
			if self.path == "/data.log":
				self.send_response(200)
				self.send_header('Content-type', 'text/plain')
				self.end_headers()
				
				path = os.path.normpath(os.path.join(os.curdir, LOGDIR));
				delimiter = str.encode('#\n')
				
				for file in os.listdir(path):
					if file.endswith(".log"):
						f = open(os.path.join(path, file), 'rb')
						self.wfile.write(str.encode(file+"\n") + f.read() + delimiter)
						f.close()
				
				return
			

			if sendReply == True:
				self.send_response(200)
				self.send_header('Content-type',mimetype)
				self.end_headers()
				
				f = open(os.curdir + os.sep + self.path, 'rb')
				self.wfile.write(f.read())
				f.close()
			else:
				self.send_error(404)
			return


		except IOError:
			self.send_error(404)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--logdir", action="store", help="directory of the log files")
	args = parser.parse_args()
	
	if args.logdir:
		LOGDIR = os.path.normpath(args.logdir)
	
		try:
			server = HTTPServer(('127.0.0.1', PORT_NUMBER), myHandler)
			print("Serving HTTP on {}:{}".format(server.socket.getsockname()[0], server.socket.getsockname()[1]))
			
			server.serve_forever()

		except KeyboardInterrupt:
			print ("Shutting down the web server")
			server.socket.close()
	
	else:
		parser.print_help()
