from gradcam import generate_gradcam_for_biomedclip
import warnings
warnings.filterwarnings("ignore")

def main():
    # Example configuration
    image_path = "/storage/projects3/e19-fyp-out-of-domain-gen-in-cv/camelyon17WILDS/patches/patient_004_node_4/patch_patient_004_node_4_x_3328_y_21792.png"
    caption = 'Tumor cells display irregular nuclei with coarse chromatin and altered CD5 expression.'
    output_dir = "./"
    saliency_layer = "blocks"  # Options: "blocks", "norm_pre", "norm", "head"
    
    print(f"Generating GradCAM for:")
    print(f"  Image: {image_path}")
    print(f"  Caption: {caption}")
    print(f"  Output directory: {output_dir}")
    print(f"  Saliency layer: {saliency_layer}")
    
    # # Generate GradCAM
    attention_map = generate_gradcam_for_biomedclip(
        image_path=image_path,
        caption=caption,
        output_dir=output_dir,
        saliency_layer=saliency_layer,
        blur=True
    )
    
    # print(f"\nGradCAM generation completed successfully!")
    # print(f"Attention map shape: {attention_map.shape}")
    # print(f"Check the '{output_dir}' directory for output images.")
        
    

if __name__ == "__main__":
    main()
