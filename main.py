import argparse
from scripts.verify.verify_model import verify_model
from scripts.optimize.model_optimizer import optimize_model
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_parser():
    parser = argparse.ArgumentParser(description='Qwen Mobile Deployment Tool')
    parser.add_argument(
        '--action',
        choices=['verify', 'optimize', 'all'],
        default='all',
        help='Action to perform'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='./models/original',
        help='Path to the model'
    )
    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    try:
        if args.action in ['verify', 'all']:
            logger.info("Starting model verification...")
            success, model, tokenizer = verify_model()
            if not success:
                logger.error("Model verification failed!")
                return
            
        if args.action in ['optimize', 'all']:
            logger.info("Starting model optimization...")
            optimize_model()
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()