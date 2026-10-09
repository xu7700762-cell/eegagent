# -*- coding: utf-8 -*-
import argparse
from .config import load_config

def main():
    parser=argparse.ArgumentParser(description='Current three-specialist EEGAgent webpage')
    parser.add_argument('--config')
    sub=parser.add_subparsers(dest='command',required=True)
    serve=sub.add_parser('serve')
    serve.add_argument('--host',default='127.0.0.1')
    serve.add_argument('--port',type=int,default=8880)
    args=parser.parse_args()
    import uvicorn
    from .service import create_app
    uvicorn.run(create_app(load_config(args.config)),host=args.host,port=args.port)

if __name__=='__main__':
    main()
