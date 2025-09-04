from BiomedCLIP_EA_Experiments.saliency_map.gradcam_util import generate_gradcam_for_biomedclip
import warnings
warnings.filterwarnings("ignore")

def main():
    # Example configuration
    image_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/patches/patient_004_node_4/patch_patient_004_node_4_x_3328_y_21792.png"
    prompt = 'This is a tumor'
    output_dir = "./"
    saliency_layer = "blocks"  
    
    print(f"Generating GradCAM for:")
    print(f"  Image: {image_path}")
    print(f"  prompt: {prompt}")
    print(f"  Output directory: {output_dir}")
    print(f"  Saliency layer: {saliency_layer}")
    
    # # Generate GradCAM
    attention_map = generate_gradcam_for_biomedclip(
        image_path=image_path,
        prompt=prompt,
        output_dir=output_dir,
        saliency_layer=saliency_layer,
        blur=True
    )
    
  
if __name__ == "__main__":
    main()
